# -*- coding: utf-8 -*-
# @Time : 2023/2/27 下午3:37
# @Author : Lingo
# @File : configuration_conica.py

from transformers.configuration_utils import PretrainedConfig


class ConicaConfig(PretrainedConfig):
    model_type = "conica"

    def __init__(
            self,
            vocab_size=49408,
            d_model=512,
            d_vision=2048,
            d_align=128,
            vision_global_pool=True,
            max_positions=320,
            position_type="sinusoid",
            vision_encoder_layers=6,
            text_encoder_layers=3,
            n_head=8,
            multimodal_decoder_layers=3,
            ffn_ratio=4,
            activation_function="gelu",
            dropout=0.1,
            droppath=0.1,
            init_std=0.02,
            use_cache=False,
            pad_token_id=0,
            queue_size=8192,
            momentum=0.995,
            bos_token_id=49406,
            eos_token_id=49407,
            is_encoder_decoder=True,
            top_k=0,
            max_length=20,
            return_dict_in_generate=True,
            seq_per_img=5,
            output_scores=True,
            do_sample=True,
            tau=0.07,
            output_hidden_states=True,
            alpha=0.5,
            vision_dropout=0.1,
            lm_dropout=0.5,
            xe_weight=1,
            co_weight=1,
            label_smoothing=0.1,
            count_similarity=True,
            pre_norm=False,
            ln_eps=1e-6,
            **kwargs
    ):
        self.vocab_size = vocab_size
        self.d_vision = d_vision
        self.d_model = d_model
        self.d_align = d_align
        self.d_ffn = d_model * ffn_ratio
        self.n_head = n_head
        self.tau = tau
        self.max_positions = max_positions
        self.vision_encoder_layers = vision_encoder_layers
        self.text_encoder_layers = text_encoder_layers
        self.multimodal_decoder_layers = multimodal_decoder_layers
        self.seq_per_img = seq_per_img
        self.lm_dropout = lm_dropout
        self.dropout = dropout
        self.droppath = droppath
        self.vision_dropout = vision_dropout
        self.activation_function = activation_function
        self.queue_size = queue_size
        self.momentum = momentum
        self.init_std = init_std
        self.position_type = position_type
        self.use_cache = use_cache
        self.alpha = alpha
        self.xe_weight = xe_weight
        self.co_weight = co_weight
        self.count_similarity = count_similarity
        self.label_smoothing = label_smoothing
        self.ln_eps = ln_eps
        self.vision_global_pool = vision_global_pool
        self.pre_norm = pre_norm
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            max_length=max_length,
            top_k=top_k,
            return_dict_in_generate=return_dict_in_generate,
            output_scores=output_scores,
            do_sample=do_sample,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
