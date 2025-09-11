# -*- coding: utf-8 -*-
# @Time : 2023/2/27 下午3:34
# @Author : Lingo
# @File : generation_conica.py
import string
import warnings
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch import nn
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import validate_stopping_criteria, GenerationMixin, SampleOutput, \
    SampleEncoderDecoderOutput, SampleDecoderOnlyOutput, BeamSearchScorer, BeamSearchOutput, \
    BeamSearchEncoderDecoderOutput, \
    BeamSearchDecoderOnlyOutput
from conica.generation_contrastive_beam_search import ContrastiveBeamSearchScorer

from transformers.generation.logits_process import ForcedEOSTokenLogitsProcessor


@dataclass
class ConicaSampleEncoderDecoderOutput(SampleEncoderDecoderOutput):
    v_embeds: Optional[torch.Tensor] = None
    t_embeds: Optional[torch.Tensor] = None
    similarity: Optional[torch.Tensor] = None

    logprobs: Optional[torch.Tensor] = None


@dataclass
class ConicaSamleDecoderOnlyOutput(SampleDecoderOnlyOutput):
    v_embeds: Optional[torch.Tensor] = None
    t_embeds: Optional[torch.Tensor] = None
    similarity: Optional[torch.Tensor] = None
    logprobs: Optional[torch.Tensor] = None


@dataclass
class ConicaBeamSearchEncoderDecoderOutput(BeamSearchEncoderDecoderOutput):
    v_embeds: Optional[torch.Tensor] = None
    t_embeds: Optional[torch.Tensor] = None
    similarity: Optional[torch.Tensor] = None
    logprobs: Optional[torch.Tensor] = None


@dataclass
class ConicaBeamSearchDecoderOnlyOutput(BeamSearchDecoderOnlyOutput):
    image_embeds: Optional[torch.Tensor] = None
    text_embeds: Optional[torch.Tensor] = None
    similarity: Optional[torch.Tensor] = None
    logprobs: Optional[torch.Tensor] = None


class ConicaGeneration(GenerationMixin):

    def sample(
            self,
            input_ids: torch.LongTensor,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            logits_warper: Optional[LogitsProcessorList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: Optional[bool] = False,
            **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]
        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        v_embeds, t_embeds = None, None
        logprobs = None
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits.clone())
            next_token_scores = logits_warper(input_ids, next_token_scores.clone())

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            _logprobs = nn.functional.log_softmax(next_token_logits, dim=-1)
            next_logprobs = _logprobs.gather(1, next_tokens.unsqueeze(1))
            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                next_logprobs = next_logprobs.masked_fill(unfinished_sequences.unsqueeze(1) == 0, 0)

            if v_embeds is None:
                v_embeds = outputs.pooler_output[0]
                t_embeds = torch.zeros((input_ids.size(0), v_embeds.size(1)), dtype=v_embeds.dtype,
                                       device=v_embeds.device)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            logprobs = torch.cat([logprobs, next_logprobs], dim=1) \
                if logprobs is not None else next_logprobs
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
                if (next_tokens == eos_token_id).any():
                    _model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                    _t_embeds = self(
                        **_model_inputs,
                        return_dict=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_unimodal_feature_only=True).pooler_output[1]
                    t_embeds[next_tokens == eos_token_id] += _t_embeds[next_tokens == eos_token_id]

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True
        if unfinished_sequences.max() != 0:
            _input_ids = torch.cat((input_ids, torch.zeros((input_ids.size(0), 1), device=input_ids.device,
                                                           dtype=torch.long).fill_(eos_token_id)), dim=1)

            _model_inputs = self.prepare_inputs_for_generation(_input_ids, **model_kwargs)
            _t_embeds = self(
                **_model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_unimodal_feature_only=True).pooler_output[1]
            # update generated ids, model inputs, and length for next step
            t_embeds[unfinished_sequences == 1] += _t_embeds[unfinished_sequences == 1]
        similarity = torch.einsum("ij,ij->i", v_embeds, t_embeds)

        if return_dict_in_generate:

            # forward pass to get next token

            if self.config.is_encoder_decoder:
                return ConicaSampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    similarity=similarity,
                    v_embeds=v_embeds,
                    t_embeds=t_embeds,
                    logprobs=logprobs
                )
            else:
                return ConicaSamleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    similarity=similarity,
                    image_embeds=v_embeds,
                    text_embeds=t_embeds,
                    logprobs=logprobs
                )
        else:
            return input_ids

    def beam_search(
            self,
            input_ids: torch.LongTensor,
            beam_scorer: BeamSearchScorer,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: Optional[bool] = False,
            **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        beam_scorer = ContrastiveBeamSearchScorer(len(beam_scorer._beam_hyps), beam_scorer.num_beams,
                                                  beam_scorer.device,
                                                  beam_scorer.length_penalty, beam_scorer.do_early_stopping,
                                                  beam_scorer.num_beam_hyps_to_keep,
                                                  beam_scorer.num_beam_groups,
                                                  alpha=self.config.alpha)
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -float("inf")
        beam_scores = beam_scores.view((batch_size * num_beams,))
        v_embeds = None
        this_peer_finished = False  # used by synced_gpus only
        logprobs = None
        count_similarity = self.config.count_similarity

        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_logits = outputs.logits[:, -1, :]
            next_logits = self.adjust_logits_during_generation(next_logits, cur_len=cur_len)
            next_logprobs = nn.functional.log_softmax(
                next_logits, dim=-1,
            )  # (batch_size * num_beams, vocab_size)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_logprobs,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            next_token_scores_processed = logits_processor(input_ids, next_logprobs.clone())

            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_logprobs)
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            next_tokens_logprob = next_logprobs.view(batch_size, -1).gather(1, next_tokens.clone())
            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")

            next_tokens = next_tokens % vocab_size
            if v_embeds is None:
                v_embeds = outputs.pooler_output[0]

            _model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            if (next_tokens == eos_token_id).any():

                _input_ids = torch.cat((input_ids, torch.zeros((batch_size * num_beams, 1), device=input_ids.device,
                                                               dtype=torch.long).fill_(eos_token_id)), dim=1)
                _model_inputs = self.prepare_inputs_for_generation(_input_ids, **_model_kwargs)
                t_embeds = self(
                    **_model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_unimodal_feature_only=True).pooler_output[1]
                similarity = torch.einsum("ij,ij->i", v_embeds, t_embeds)
            else:
                similarity = None
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                logprobs,
                next_tokens_logprob,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                similarity=similarity,
                count_similarity=count_similarity
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]
            beam_logprobs = beam_outputs["next_beam_logprobs"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            logprobs = beam_logprobs.unsqueeze(-1) if logprobs is None else torch.cat(
                [logprobs[beam_idx, :], beam_logprobs.unsqueeze(-1)], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)
            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))
            # increase cur_len

            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True
        if not beam_scorer.is_done:
            _model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            _input_ids = torch.cat((input_ids, torch.zeros((batch_size * num_beams, 1), device=input_ids.device,
                                                           dtype=torch.long).fill_(eos_token_id)), dim=1)
            _model_inputs = self.prepare_inputs_for_generation(_input_ids, **_model_kwargs)
            t_embeds = self(
                **_model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_unimodal_feature_only=True).pooler_output[1]
            similarity = torch.einsum("ij,ij->i", v_embeds, t_embeds)
        else:
            similarity = None
        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            logprobs,
            next_tokens_logprob,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            similarity=similarity,
            count_similarity=count_similarity
        )
        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return ConicaBeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    similarity=sequence_outputs["similarity"],
                    v_embeds=v_embeds,
                    t_embeds=t_embeds,
                    logprobs=sequence_outputs["logprobs"],
                )
            else:
                return ConicaBeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    similarity=sequence_outputs["similarity"],
                    v_embeds=v_embeds,
                    t_embeds=t_embeds,
                )
        else:
            return sequence_outputs["sequences"]