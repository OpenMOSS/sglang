from typing import List, Union

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import get_global_server_args


class AdditionalSampleProcessMixin:
    def __init__(self):
        model_arches = ("MossTTSDWithCodec", "MossTTSDForCausalLM")
        self._postprocess_dispatcher = {
            arch: self.moss_ttsd_postprocess_tokens for arch in model_arches
        }
        self._preprocess_dispatcher = {
            arch: self.moss_ttsd_preprocess_logits for arch in model_arches
        }

    def preprocess_logits(
        self,
        logits_output: Union[LogitsProcessorOutput, List[LogitsProcessorOutput]],
        sampling_info: SamplingBatchInfo,
        current_generation_step: torch.Tensor,
    ):
        if get_global_server_args().delay_pattern:
            model_arch = self.model_config.hf_config.architectures[0]
            preprocess_fn = self._preprocess_dispatcher.get(model_arch)
            if preprocess_fn is not None:
                return preprocess_fn(
                    logits_output,
                    sampling_info,
                    current_generation_step,
                )

    def _postprocess_tokens(
        self,
        next_token_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if get_global_server_args().delay_pattern:
            model_arch = self.model_config.hf_config.architectures[0]
            postprocess_fn = self._postprocess_dispatcher.get(model_arch)
            if postprocess_fn is not None:
                (
                    forward_batch.current_generation_step,
                    forward_batch.needs_additional_steps,
                    forward_batch.unfinished_sequences,
                ) = postprocess_fn(
                    next_token_ids,
                    forward_batch.current_generation_step,
                    forward_batch.truncated_input_ids,
                    forward_batch.needs_additional_steps,
                    forward_batch.unfinished_sequences,
                )

    def is_speech_token(self, token_id: torch.Tensor) -> torch.Tensor:
        return (token_id <= self.model_config.hf_text_config.speech_token_range[1]) & (
            token_id >= self.model_config.hf_text_config.speech_token_range[0]
        )

    def moss_ttsd_preprocess_logits(
        self,
        logits_output: Union[LogitsProcessorOutput, List[LogitsProcessorOutput]],
        sampling_info: SamplingBatchInfo,
        current_generation_step: torch.Tensor,
    ):
        steps = current_generation_step
        if steps.device != logits_output[0].next_token_logits.device:
            steps = steps.to(device=logits_output[0].next_token_logits.device)
        if (
            steps.dim() != 1
            or steps.shape[0] != logits_output[0].next_token_logits.shape[0]
        ):
            raise ValueError(
                "current_generation_step must be a 1D tensor of shape [batch]. "
                f"Got {tuple(steps.shape)} with batch={logits_output[0].next_token_logits.shape[0]}"
            )

        for i, logits in enumerate(logits_output):
            pad_id = self.model_config.hf_config.pad_token[i]
            if i != 0:
                row_mask = (steps + 1) > i
                if torch.any(row_mask):
                    logits.next_token_logits[row_mask, pad_id] = -torch.inf
            else:
                row_mask = steps <= (self.model_config.channels - 1)
                if torch.any(row_mask):
                    logits.next_token_logits[row_mask, pad_id] = -torch.inf
        return logits_output

    def moss_ttsd_postprocess_tokens(
        self,
        next_token_ids: torch.Tensor,
        current_generation_step: torch.Tensor,
        truncated_input_ids: torch.Tensor,
        needs_additional_steps: torch.Tensor,
        unfinished_sequences: torch.Tensor,
    ):
        channels = self.model_config.channels
        indices = (~self.is_speech_token(next_token_ids[:, 0])) & (
            needs_additional_steps < 0
        )
        needs_additional_steps[indices] = channels - 1

        # current_generation_step is a per-sample tensor of shape [batch].
        # truncated_input_ids is expected to be [batch, channels - 1, channels].
        # For each sample b with step s < channels - 1, we fill positions (s+1 .. end)
        # from truncated_input_ids[b, s, :].
        steps = current_generation_step
        if steps.device != next_token_ids.device:
            steps = steps.to(device=next_token_ids.device)
        if steps.dim() != 1 or steps.shape[0] != next_token_ids.shape[0]:
            raise ValueError(
                "current_generation_step must be a 1D tensor of shape [batch]. "
                f"Got {tuple(steps.shape)} with batch={next_token_ids.shape[0]}"
            )

        valid = steps < (channels - 1)
        # Additional guard: if this sample's truncated_input_ids are all zeros,
        # treat it as invalid for delay-pattern filling.
        if (
            isinstance(truncated_input_ids, torch.Tensor)
            and truncated_input_ids.dim() >= 2
            and truncated_input_ids.shape[0] == next_token_ids.shape[0]
        ):
            all_zero = (truncated_input_ids == 0).flatten(start_dim=1).all(dim=1)
            if all_zero.device != valid.device:
                all_zero = all_zero.to(device=valid.device)
            valid = valid & (~all_zero)
        if torch.any(valid):
            truncated = truncated_input_ids
            if truncated.device != next_token_ids.device:
                truncated = truncated.to(device=next_token_ids.device)
            if (
                truncated.dim() == 3
                and truncated.shape[0] == next_token_ids.shape[0]
                and truncated.shape[1] == channels - 1
                and truncated.shape[2] == channels
            ):
                steps_idx = steps.to(torch.long).clamp(min=0, max=channels - 2)
                batch_idx = torch.arange(
                    next_token_ids.shape[0], device=next_token_ids.device
                )
                gathered = truncated[batch_idx, steps_idx, :].to(
                    dtype=next_token_ids.dtype
                )
                pos = torch.arange(channels, device=next_token_ids.device).unsqueeze(0)
                fill_mask = valid.unsqueeze(1) & (pos > steps.unsqueeze(1))
                next_token_ids[fill_mask] = gathered[fill_mask]
            else:
                # Fallback: use pad tokens (per-channel), only for positions beyond step.
                pos = torch.arange(channels, device=next_token_ids.device).unsqueeze(0)
                fill_mask = valid.unsqueeze(1) & (pos > steps.unsqueeze(1))
                pad_row = torch.tensor(
                    self.model_config.hf_config.pad_token,
                    dtype=next_token_ids.dtype,
                    device=next_token_ids.device,
                ).unsqueeze(0)
                next_token_ids[fill_mask] = pad_row.expand_as(next_token_ids)[fill_mask]

        # Handle tensor boolean operations properly
        mask = (needs_additional_steps > 0) & (needs_additional_steps < channels - 1)
        if torch.any(mask):
            next_token_ids[mask, 0] = next(iter(self.model_config.hf_eos_token_id))
            for i in range(1, channels):
                channel_mask = mask & (needs_additional_steps < channels - i)
                if torch.any(channel_mask):
                    next_token_ids[channel_mask, i] = (
                        self.model_config.hf_config.pad_token[i]
                    )

        for i in range(channels):
            pddp = (
                next(iter(self.model_config.hf_eos_token_id))
                if i == 0
                else self.model_config.hf_config.pad_token[i]
            )
            next_token_ids[:, i] = next_token_ids[
                :, i
            ] * unfinished_sequences + pddp * (1 - unfinished_sequences)

        needs_additional_steps = torch.where(
            needs_additional_steps > 0,
            needs_additional_steps - 1,
            needs_additional_steps,
        )

        stopping = (
            next_token_ids[:, 0] == self.model_config.hf_config.pad_token[0]
        ) | (needs_additional_steps == 0)
        unfinished_sequences = unfinished_sequences & ~stopping
        unfinished_sequences = unfinished_sequences | (needs_additional_steps > 0)
        current_generation_step += 1
        return (
            current_generation_step,
            needs_additional_steps,
            unfinished_sequences,
        )
