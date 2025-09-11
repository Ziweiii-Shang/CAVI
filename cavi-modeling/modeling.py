# -*- coding: utf-8 -*-
# @Time : 2023/2/27 下午3:11
# @Author : Lingo
# @File : modeling_conica.py
import copy
from transformers.modeling_outputs import BaseModelOutputWithPooling, \
    BaseModelOutputWithPoolingAndCrossAttentions, Seq2SeqModelOutput, Seq2SeqLMOutput
from torch.distributed import get_world_size, all_gather
from .configuration_conica import ConicaConfig
from typing import Optional, Tuple
from torch import nn
import torch.utils.checkpoint
from transformers.activations import ACT2FN
from torch.nn.functional import softmax, dropout, cross_entropy, normalize
from transformers.utils import logging
from transformers import PreTrainedModel
import torch
import os
import torch.nn.functional as F
from .generation_conica import ConicaGeneration

logger = logging.get_logger(__name__)
from dataclasses import dataclass


@dataclass
class Seq2SeqModelOutputWithPooling(Seq2SeqModelOutput):
    pooler_output: Tuple[torch.FloatTensor] = None


@dataclass()
class Seq2SeqLMOutputWithPooling(Seq2SeqLMOutput):
    pooler_output: Tuple[torch.FloatTensor] = None


def droppath(x, p: float = 0., training: bool = False, scale_by_keep=True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if p == 0. or not training:
        return x
    keep_prob = 1 - p
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


class SinusoidEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        dim = torch.arange(d_model // 2, dtype=torch.float32).unsqueeze(0) / d_model
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        sin = torch.sin(pos / 10000 ** (2 * dim))
        cos = torch.cos(pos / 10000 ** (2 * dim))
        pos_encoding = torch.zeros(max_len, d_model)

        pos_encoding[:, ::2] = sin
        pos_encoding[:, 1::2] = cos
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        positions = self.pos_encoding[past_key_values_length:past_key_values_length + seq_len].unsqueeze(0).repeat(bsz,
                                                                                                                   1, 1)
        print(f"past_key_values_length: {past_key_values_length}, seq_len: {seq_len}")
        print(f"pos_encoding.shape: {self.pos_encoding.shape}")

        return positions


class ConicaLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, max_position: int, d_modal: int):
        super().__init__(max_position, d_modal)
        self.max_position = max_position

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""

        bsz, seq_len = input_ids.shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long,
            device=self.weight.device
        ).unsqueeze(0).repeat(bsz, 1)
        return super().forward(positions)


class ConicaVisionEmbedding(nn.Module):
    def __init__(self, config: ConicaConfig):
        super().__init__()
        self.vision_embeds = nn.Linear(config.d_vision,config.d_model)
        self.vision_act = ACT2FN[config.activation_function]
        self.vision_norm = nn.LayerNorm(config.d_model,config.ln_eps)
        self.dropout = config.vision_dropout

    def forward(self, vision_feats):
        hidden_states = self.vision_embeds(vision_feats)
        hidden_states = self.vision_act(hidden_states)
        hidden_states = self.vision_norm(hidden_states)
        return dropout(hidden_states, p=self.dropout, training=self.training)

class ConicaAttention(nn.Module):
    def __init__(self, config: ConicaConfig):
        super().__init__()
        d_modal = config.d_model
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_head = self.d_model // self.n_head
        self.scale = self.d_head ** -0.5
        self.dropout = config.dropout

        self.q_proj = nn.Linear(d_modal, d_modal)
        self.k_proj = nn.Linear(d_modal, d_modal)
        self.v_proj = nn.Linear(d_modal, d_modal)
        self.out_proj = nn.Linear(d_modal, d_modal)
        self.is_decoder = config.is_decoder

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.n_head, self.d_head).transpose(1, 2).contiguous()
        #return tensor.view(bsz, 1, self.n_head, self.d_head)
    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ):
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # decoder cross_attention
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # decoder cross_attention
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # decoder self_attention
            # reuse k, v,
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # encoder/decoder self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)
        proj_shape = (bsz * self.n_head, -1, self.d_head)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.n_head, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.n_head, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.n_head, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.n_head, tgt_len, src_len)

        attn_weights = softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.n_head,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.n_head,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.n_head, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.n_head, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.n_head, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.n_head, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.n_head, tgt_len, self.d_head):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.n_head, tgt_len, self.d_head)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.n_head, tgt_len, self.d_head)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can
        # be partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.d_model)

        attn_output = self.out_proj(attn_output)

        attn_output = dropout(attn_output, p=self.dropout, training=self.training)

        return attn_output, attn_weights_reshaped, past_key_value


class ConicaFFN(nn.Module):
    def __init__(self, config: ConicaConfig):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.activation_function]
        self.in_proj = nn.Linear(config.d_model, config.d_ffn)
        self.out_proj = nn.Linear(config.d_ffn, config.d_model)
        self.dropout = self.config.dropout

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.in_proj(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.out_proj(hidden_states)
        hidden_states = dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states


class ConicaLayer(nn.Module):
    def __init__(self, config: ConicaConfig, droppath=0., cross_attention=False):
        super().__init__()
        self.config = config
        self.self_attn = ConicaAttention(config)
        self.self_attn_norm = nn.LayerNorm(config.d_model, config.ln_eps)
        self.cross_attn = None
        if cross_attention:
            self.cross_attn = ConicaAttention(config)
            self.cross_attn_norm = nn.LayerNorm(config.d_model, config.ln_eps)
        self.ffn = ConicaFFN(config)
        self.ffn_norm = nn.LayerNorm(config.d_model, config.ln_eps)
        self.droppath = droppath
        self.pre_norm = config.pre_norm

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ):
        self_attn_weights, cross_attn_weights = None, None
        residual = hidden_states
        if self.pre_norm:
            hidden_states = self.self_attn_norm(hidden_states)
        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        hidden_states, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value
        )
        hidden_states = droppath(hidden_states, self.droppath, self.training)
        hidden_states = residual + hidden_states
        if not self.pre_norm:
            hidden_states = self.self_attn_norm(hidden_states)
        if self.cross_attn is not None and encoder_hidden_states is not None:
            residual = hidden_states
            if self.pre_norm:
                hidden_states = self.cross_attn_norm(hidden_states)
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.cross_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = droppath(hidden_states, self.droppath, self.training)
            hidden_states = residual + hidden_states
            if not self.pre_norm:
                hidden_states = self.cross_attn_norm(hidden_states)
            present_key_value = present_key_value + cross_attn_present_key_value

            # add cross-attn to positions 3,4 of present_key_value tuple
        residual = hidden_states
        if self.pre_norm:
            hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = droppath(hidden_states, self.droppath, self.training)
        hidden_states = residual + hidden_states
        if not self.pre_norm:
            hidden_states = self.ffn_norm(hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


        


class ConicaPreTrainedModel(PreTrainedModel):
    config_class = ConicaConfig
    base_model_prefix = "conica"
    main_input_name = "vision_feats"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_unexpected = None

    def _init_weights(self, module):
        std = self.config.init_std
        # d_model = self.config.d_model
        # attn_std = (d_model ** -0.5)
        # ffn_std = (2 * d_model) ** -0.5
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (ConicaVisionEncoder, ConicaTextDecoder)):
            module.gradient_checkpointing = value


class ConicaVisionEncoder(ConicaPreTrainedModel):
    def __init__(self, config: ConicaConfig):
        super().__init__(config)
        self.embed_vision = ConicaVisionEmbedding(config)
        self.global_pool = config.vision_global_pool
        dpr = [x.item() for x in
               torch.linspace(0, config.droppath, config.vision_encoder_layers)]  # stochastic depth decay rule
        self.layers = nn.ModuleList([ConicaLayer(config, dpr[i], False) for i in range(config.vision_encoder_layers)])
        self.gradient_checkpointing = False
        self.vision_proj = nn.Linear(config.d_model, config.d_align)
        self.final_norm = nn.LayerNorm(config.d_model, config.ln_eps) if config.pre_norm else nn.Identity()
        self.post_init()

    def forward(
            self,
            vision_feats=None,
            attention_mask=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # expand attention_mask

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        # check if head_mask has a correct number of layers specified if desired\

        hidden_states = self.embed_vision(vision_feats)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attention_mask = _expand_mask(attention_mask, vision_feats.dtype)
        else:
            expanded_attention_mask = None
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                ####测试
                #print(f"Layer {idx} - all_hidden_states length: {len(all_hidden_states)}")  # 打印每一层后的 all_hidden_states 长度
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    expanded_attention_mask,
                    (head_mask[idx] if head_mask is not None else None)
                )
            else:
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    attention_mask=expanded_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                    past_key_value=None,
                    use_cache=False,
                )
            hidden_states = layer_outputs[0]
            #print('打印每一层的encoder输出：')
            #print('###################################################')
            #print(f"Encoder layer {idx} output shape: {hidden_states.shape}")
            #print(f"Encoder layer {idx} output: {hidden_states}")
            #print('###################################################')

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        #修改储存每层隐藏态#额外改动，对比层数是否有问题
        if output_hidden_states:
            all_hidden_states = all_hidden_states[:len(self.layers)]
            #####测试
            print(f"Final all_hidden_states length: {len(all_hidden_states)}")  # 打印最终的 all_hidden_states 长度
        hidden_states = self.final_norm(hidden_states)
        if self.global_pool:
            pooler_output = self.vision_proj((hidden_states * attention_mask.unsqueeze(-1)).sum(1)
                                             / attention_mask.sum(-1).unsqueeze(-1))
        else:
            pooler_output = self.vision_proj(hidden_states[:, 0, :])
        pooler_output = normalize(pooler_output, dim=-1)
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions, pooler_output])
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            pooler_output=pooler_output
        )


class ConicaTextDecoder(ConicaPreTrainedModel):
    def __init__(self, config: ConicaConfig):
        super().__init__(config)
        self.embed_token = nn.Embedding(config.vocab_size, config.d_model, config.pad_token_id)
        self.embed_scale = config.d_model ** 0.5 if config.position_type == "sinusoid" else 1
        self.embed_positions = SinusoidEncoding(config.max_positions,
                                                config.d_model) if config.position_type == "sinusoid" else \
            ConicaLearnedPositionalEmbedding(config.max_positions + 1, config.d_model)

        dpr_1 = [x.item() for x in
                 torch.linspace(0, config.droppath, config.text_encoder_layers)]  # stochastic depth decay rule
        dpr_2 = [x.item() for x in
                 torch.linspace(0, config.droppath, config.text_encoder_layers)]  # stochastic depth decay rule

        self.layers = nn.ModuleList([ConicaLayer(config, dpr_1[i], False) for i in range(config.text_encoder_layers)] +
                                    [ConicaLayer(config, dpr_2[i], True) for i in
                                     range(config.multimodal_decoder_layers)])
        self.text_proj = nn.Linear(config.d_model, config.d_align)
        self.text_norm = nn.LayerNorm(config.d_model, config.ln_eps) if config.pre_norm else nn.Identity()
        self.dropout = config.dropout
        self.gradient_checkpointing = False
        self.final_norm = nn.LayerNorm(config.d_model, config.ln_eps) if config.pre_norm else nn.Identity()
        self.post_init()

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]

            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            #新增参数来接收encoder每层的输出
            encoder_all_hidden_states=None,
            encoder_attention_mask=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            return_unimodal_feature_only: bool = False
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        ###########################################测试
        if input_ids is not None:
            print(f"Decoder input_ids shape: {input_ids.shape}")
        if inputs_embeds is not None:
            print(f"Decoder inputs_embeds shape: {inputs_embeds.shape}")




        # retrieve input_ids and inputs_embeds
        if input_ids is None:
            raise ValueError("You have to specify decoder_input_ids and decoder_inputs_embeds at the same time")
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_token(input_ids) * self.embed_scale
        if encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
        positions = self.embed_positions(input_ids, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = dropout(hidden_states, p=self.dropout, training=self.training)
        #######################################测试
        print(f"Initial hidden_states shape: {hidden_states.shape}")  
        #########################################
        
        
        
        
        
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None
        pooler_output = None
        if encoder_all_hidden_states is not None:
            print(f"Received encoder_all_hidden_states length: {len(encoder_all_hidden_states)}")
            if len(encoder_all_hidden_states) != len(self.layers):
                raise ValueError(f"Expected {len(self.layers)} encoder layers, but got {len(encoder_all_hidden_states)} encoder hidden states.")

        #Verify encoder_all_hidden_states length matches the number of encoder layers
        #if encoder_all_hidden_states is not None:
        #    if len(encoder_all_hidden_states) != len(self.layers):
        #        raise ValueError(f"Expected {len(self.layers)} encoder layers, but got {len(encoder_all_hidden_states)} encoder hidden states.")

        # Check the shape of encoder_all_hidden_states for each layer
        #if encoder_all_hidden_states is not None:
        #    for idx, enc_hidden in enumerate(encoder_all_hidden_states):
        #        if enc_hidden.shape[2] != hidden_states.shape[2]:
        #            raise ValueError(f"Encoder hidden state layer {idx} has incompatible hidden dimension size {enc_hidden.shape[2]}. Expected {hidden_states.shape[2]}.")


        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for idx, layer in enumerate(self.layers):

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    encoder_hidden_states,
                    cross_attn_head_mask[idx] if cross_attn_head_mask else None,
                    encoder_attention_mask,
                    None,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]
            print(f"Layer {idx} output hidden_states shape: {hidden_states.shape}")
            ################################
            print(' every encoder in cross-att')  
             # 处理每层 encoder 输出的 cross-attention
            if output_attentions and encoder_all_hidden_states is not None:
                cross_attention_outputs = []
                for enc_idx in range(len(encoder_all_hidden_states)):
                    cross_attention_output = self.layers[idx].cross_attn(
                        hidden_states,
                        encoder_all_hidden_states[enc_idx],
                        encoder_all_hidden_states[enc_idx],
                        encoder_attention_mask
                    )
                    cross_attention_outputs.append(cross_attention_output)

                avg_cross_attention_output = torch.stack(cross_attention_outputs).mean(dim=0)
                print(f"Avg cross-attention output at decoder layer {idx}: {avg_cross_attention_output.shape}")


                # 替代原始 hidden_states 为新计算的 cross-attention 输出
                hidden_states = avg_cross_attention_output   
            ####################################


            print(f"Layer {idx} output hidden_states shape: {hidden_states.shape}")
            if use_cache and layer_outputs[3 if output_attentions else 1] is not None:
                print(f"Layer {idx} cached key/value shapes: {[x.shape for x in layer_outputs[3 if output_attentions else 1]]}")
################################################

            

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)
            if idx == self.config.text_encoder_layers - 1:
                pooler_output = self.text_proj(
                    self.text_norm(hidden_states[torch.arange(input_ids.size(0)), torch.argmax(input_ids, dim=1)]))
                pooler_output = normalize(pooler_output, dim=-1)
                if return_unimodal_feature_only:
                    break
        ########################################
        print(f"Final hidden_states shape: {hidden_states.shape}")
        # add hidden states from the last decoder layer
        hidden_states = self.final_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in
                [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions, pooler_output]
            )
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
            pooler_output=pooler_output
        )

######################################整个model类修改了，如果有问题重新去找源代码


class CONICAModel(ConicaPreTrainedModel):
    def __init__(self, config: ConicaConfig):
        super().__init__(config)
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = ConicaVisionEncoder(encoder_config)
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        self.decoder = ConicaTextDecoder(decoder_config)
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.decoder.embed_token

    def set_input_embeddings(self, new_embeddings):
        self.decoder.embed_token = new_embeddings

    def forward(
            self,
            vision_feats,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            return_unimodal_feature_only=False
    ):
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            raise ValueError(
                "`decoder_input_ids` and `decoder_inputs_embeds` "
                "can not be None at the same time "
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        expand_size = 1

        if encoder_outputs is None:
            if decoder_input_ids is None:
                expand_size = decoder_inputs_embeds.size(0) // vision_feats.size(0)
            elif decoder_inputs_embeds is None:
                expand_size = decoder_input_ids.size(0) // vision_feats.size(0)
            encoder_outputs = self.encoder(
                vision_feats,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                ##改
                output_hidden_states=True,
                return_dict=True,  # Ensures we get a dict with all required fields
            )

            expanded_return_idx = torch.arange(vision_feats.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(
                vision_feats.device)
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutputWithPooling):
            encoder_outputs = BaseModelOutputWithPooling(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                pooler_output=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
            )

        encoder_hidden_states = encoder_outputs.last_hidden_state.index_select(0, expanded_return_idx.to(
            encoder_outputs.last_hidden_state.device)) if expand_size > 1 else encoder_outputs.last_hidden_state
        if attention_mask is not None:
            attention_mask = attention_mask.index_select(0, expanded_return_idx.to(
                encoder_outputs.last_hidden_state.device)) if expand_size > 1 else attention_mask
        if not self.config.vision_global_pool and self.training:
            encoder_hidden_states = encoder_hidden_states[:, 1:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, 1:]
        ##############改
        if len(encoder_outputs.hidden_states) != len(self.decoder.layers):
            raise ValueError(f"Encoder hidden states ({len(encoder_outputs.hidden_states)}) do not match Decoder layers ({len(self.decoder.layers)})")

        print(f"Encoder outputs hidden_states length: {len(encoder_outputs.hidden_states)}")  # 打印encoder的hidden_states长度

        # Pass all hidden states to decoder
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            #改
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_all_hidden_states=encoder_outputs.hidden_states,  # Pass the list of encoder hidden states
            encoder_attention_mask=attention_mask if attention_mask is not None else None,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            return_unimodal_feature_only=return_unimodal_feature_only
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutputWithPooling(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            pooler_output=(encoder_outputs.pooler_output, decoder_outputs.pooler_output)
        )

#######进行修改
class ConicaModelWithLMHead(ConicaPreTrainedModel, ConicaGeneration):
    def __init__(self, config: ConicaConfig):
        super().__init__(config)
        self.model = CONICAModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.tau = nn.Parameter(torch.ones([]) * config.tau)
        self.post_init()
        config_momentum = copy.deepcopy(config)
        config_momentum.multimodal_decoder_layers = 0
        self.model_m = CONICAModel(config_momentum)

        self.copy_params()

        self.register_buffer("v_queue", torch.randn(config.queue_size, config.d_align))
##############################################################################################################       
        self.register_buffer("t_queue", torch.randn(config.queue_size * config.seq_per_img, config.d_align))

        self.v_queue = nn.functional.normalize(self.v_queue, dim=1)
        self.t_queue = nn.functional.normalize(self.t_queue, dim=1)
        self.dropout = config.lm_dropout

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.model.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def copy_params(self):
        for name_m, parameter_m in self.model_m.named_parameters():
            parameter_m.data.copy_(self.model.state_dict()[name_m].data)
            parameter_m.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for name_m, parameter_m in self.model_m.named_parameters():
            parameter = self.model.state_dict()[name_m]
            parameter_m.data = parameter_m.data * self.config.momentum + parameter.data * (1 - self.config.momentum)

    @staticmethod
    @torch.no_grad()
    def concat_all_gather(tensor):
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank == -1:
            return tensor
        tensors_gather = [torch.ones_like(tensor) for _ in range(get_world_size())]
        all_gather(tensors_gather, tensor, async_op=False)
        return torch.cat(tensors_gather, dim=0)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, v_embeds, t_embeds):
        v_embeds = self.concat_all_gather(v_embeds)
        t_embeds = self.concat_all_gather(t_embeds)

        v_embeds = torch.cat((v_embeds, self.v_queue.clone().detach()), dim=0)
        t_embeds = torch.cat((t_embeds, self.t_queue.clone().detach()), dim=0)
        self.v_queue = v_embeds[:len(self.v_queue)]
        self.t_queue = t_embeds[:len(self.t_queue)]

    def prepare_inputs_for_generation(
            self,
            input_ids,
            vision_feats=None,
            past_key_values=None,
            attention_mask=None,
            head_mask=None,
            decoder_attention_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "vision_feats": vision_feats,  # 修改：确保传递视觉特征
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "decoder_attention_mask": None,
            "use_cache": use_cache,
        }

    def forward(self, *args, **kwargs):
        is_generate = kwargs.get('is_generate', False)
        if is_generate:
            kwargs.pop("is_generate")
            return super().generate.__wrapped__(self, *args, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    def init_tau(self):
        nn.init.constant_(self.tau.data, self.config.tau)

    def _forward(
            self,
            vision_feats=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            return_unimodal_feature_only=False
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = labels
        #################################################################################
        if decoder_input_ids is not None:
            print(f"Forward decoder_input_ids shape: {decoder_input_ids.shape}")
        if vision_feats is not None:
            print(f"Forward vision_feats shape: {vision_feats.shape}")






        outputs = self.model(
            vision_feats=vision_feats,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            return_unimodal_feature_only=return_unimodal_feature_only
        )
##########################################
        print(f"Model outputs last_hidden_state shape: {outputs.last_hidden_state.shape}")
        ################################


        loss = None
        lm_logits = None

        if not return_unimodal_feature_only:
            lm_logits = self.lm_head(
                nn.functional.dropout(outputs.last_hidden_state, self.dropout, self.training)
            )

        v_embeds, t_embeds = outputs.pooler_output[0], outputs.pooler_output[1]

        if labels is not None:
            with torch.no_grad():
                self.tau.clamp_(min=0.01, max=0.5)
                self._momentum_update()

                outputs_m = self.model_m(
                    vision_feats=vision_feats,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    head_mask=head_mask,
                    decoder_head_mask=decoder_head_mask,
                    cross_attn_head_mask=cross_attn_head_mask,
                    encoder_outputs=encoder_outputs,
                    past_key_values=past_key_values,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    return_unimodal_feature_only=True
                )

                v_embeds_m, t_embeds_m = outputs_m.pooler_output[0], outputs_m.pooler_output[1]
                v_embeds_all = torch.cat([v_embeds_m, self.v_queue.clone().detach()], dim=0)
                t_embeds_all = torch.cat([t_embeds_m, self.t_queue.clone().detach()], dim=0)

                expand_size = 1
                if decoder_input_ids is None:
                    expand_size = decoder_inputs_embeds.size(0) // vision_feats.size(0)
                elif decoder_inputs_embeds is None:
                    expand_size = decoder_input_ids.size(0) // vision_feats.size(0)

            sim_i2t = torch.div(torch.matmul(v_embeds, t_embeds_all.t()), self.tau)
            sim_t2i = torch.div(torch.matmul(t_embeds, v_embeds_all.t()), self.tau)
            sim_i2t_target = torch.zeros_like(sim_i2t, device=sim_i2t.device)
            sim_t2i_target = torch.zeros_like(sim_t2i, device=sim_t2i.device)
            for i in range(len(sim_i2t)):
                sim_i2t_target[i, i * expand_size:(i + 1) * expand_size] = 1 / expand_size
                sim_t2i_target[i * expand_size:(i + 1) * expand_size, i] = 1

            co_loss = (cross_entropy(sim_i2t, sim_i2t_target, label_smoothing=self.model.config.label_smoothing) +
                       cross_entropy(sim_t2i, sim_t2i_target, label_smoothing=self.model.config.label_smoothing)) / 2 * self.config.co_weight

            self._dequeue_and_enqueue(v_embeds_m, t_embeds_m)
            loss = co_loss

            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss = cross_entropy(shift_logits.view(-1, self.config.vocab_size),
                                    shift_labels.view(-1),
                                    ignore_index=self.config.pad_token_id,
                                    label_smoothing=self.model.config.label_smoothing) * self.config.xe_weight
            loss += lm_loss

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
gpt的decoder：
class ConicaTextDecoder(ConicaPreTrainedModel):
    def __init__(self, config: ConicaConfig):
        super().__init__(config)
        self.embed_token = nn.Embedding(config.vocab_size, config.d_model, config.pad_token_id)
        self.embed_scale = config.d_model ** 0.5 if config.position_type == "sinusoid" else 1
        self.embed_positions = SinusoidEncoding(config.max_positions,
                                                config.d_model) if config.position_type == "sinusoid" else \
            ConicaLearnedPositionalEmbedding(config.max_positions + 1, config.d_model)

        dpr_1 = [x.item() for x in
                 torch.linspace(0, config.droppath, config.text_encoder_layers)]  # stochastic depth decay rule
        dpr_2 = [x.item() for x in
                 torch.linspace(0, config.droppath, config.text_encoder_layers)]  # stochastic depth decay rule

        self.layers = nn.ModuleList([ConicaLayer(config, dpr_1[i], False) for i in range(config.text_encoder_layers)] +
                                    [ConicaLayer(config, dpr_2[i], True) for i in
                                     range(config.multimodal_decoder_layers)])
        self.text_proj = nn.Linear(config.d_model, config.d_align)
        self.text_norm = nn.LayerNorm(config.d_model, config.ln_eps) if config.pre_norm else nn.Identity()
        self.dropout = config.dropout
        self.gradient_checkpointing = False
        self.final_norm = nn.LayerNorm(config.d_model, config.ln_eps) if config.pre_norm else nn.Identity()
        self.post_init()

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]

            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            #######创建接收参数#######
            encoder_all_hidden_states=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            return_unimodal_feature_only: bool = False
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is None:
            raise ValueError("You have to specify decoder_input_ids and decoder_inputs_embeds at the same time")
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_token(input_ids) * self.embed_scale
        if encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
        positions = self.embed_positions(input_ids, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = dropout(hidden_states, p=self.dropout, training=self.training)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None
        pooler_output = None
        cross_attention_outputs = []
        ###########################确保接收encoder每层输出#####################
        if encoder_all_hidden_states is None:
            raise ValueError("encoder_all_hidden_states must be provided for cross-attention.")
        if len(encoder_all_hidden_states) != self.config.vision_encoder_layers:
            raise ValueError(f"Expected {self.config.vision_encoder_layers} layers for encoder_all_hidden_states, but got {len(encoder_all_hidden_states)} layers.")
        #######遍历解码器层#####


        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for idx, layer in enumerate(self.layers):

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    encoder_hidden_states,
                    cross_attn_head_mask[idx] if cross_attn_head_mask else None,
                    encoder_attention_mask,
                    None,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    #####第一次改动:将encoder_all_hidden_states的id逐步传入
                    encoder_hidden_states=encoder_all_hidden_states[idx] if encoder_all_hidden_states is not None else None,  # 使用每一层编码器的输出
                    #encoder_hidden_states=encoder_hidden_states,##传入encoder的最后一层
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]###decoder_hidden_states的更新
            #######第二次改动，收集cross-att输出
            if encoder_all_hidden_states is not None and len(layer_outputs) > 2:
                cross_attention_outputs.append(layer_outputs[2])##hiddenstates接收cross-attention输出

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                ###交叉注意力机制的注意力权重
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)
            if idx == self.config.text_encoder_layers - 1:
                pooler_output = self.text_proj(
                    self.text_norm(hidden_states[torch.arange(input_ids.size(0)), torch.argmax(input_ids, dim=1)]))
                pooler_output = normalize(pooler_output, dim=-1)
                if return_unimodal_feature_only:
                    break
       
       
        # 计算交叉注意力的平均值
        if cross_attention_outputs:
            average_cross_attention = sum(cross_attention_outputs) / len(cross_attention_outputs)
            all_cross_attentions = (average_cross_attention,)  # 更新all_cross_attentions为平均值

        # add hidden states from the last decoder layer
        hidden_states = self.final_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in
                [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions, pooler_output]
            )
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            ######交叉注意力######
            cross_attentions=all_cross_attentions,
            pooler_output=pooler_output
        )