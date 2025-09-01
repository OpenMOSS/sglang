"""
MOSS-TTSD Multimodal Processor for end-to-end text-to-speech synthesis.
Handles preprocessing (text/audio to tokens) and postprocessing (tokens to audio).
"""

import base64
import io
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from transformers.models.moss_ttsd.processing_moss_ttsd import MossTTSDProcessor

from sglang.srt.models.moss_ttsd import MossTTSDForCausalLM
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.utils import logger


class MossTTSDMultimodalProcessor(BaseMultimodalProcessor):
    """Multimodal processor for MOSS-TTSD model."""

    models = [MossTTSDForCausalLM]

    def __init__(
        self,
        hf_config,
        server_args,
        _processor,
        transport_mode="vllm_2_sglang",
        *args,
        **kwargs,
    ):
        """Initialize the MOSS-TTSD processor.

        Args:
            hf_config: HuggingFace model configuration
            server_args: Server arguments including xy_tokenizer_path
            _processor: The underlying MossTTSDProcessor instance
            transport_mode: Transport mode for multimodal data
        """
        super().__init__(
            hf_config, server_args, _processor, transport_mode, *args, **kwargs
        )

        self.model_path = server_args.model_path
        self.xy_tokenizer_path = server_args.xy_tokenizer_path
        self.max_channels = getattr(hf_config, "channels", 8)

        if not self.xy_tokenizer_path:
            raise ValueError("xy_tokenizer_path is required for MOSS-TTSD processor")

        if not os.path.exists(self.xy_tokenizer_path):
            raise ValueError(
                f"XY tokenizer path does not exist: {self.xy_tokenizer_path}"
            )

        # Read sample rate from XY tokenizer config
        self.input_sample_rate, self.output_sample_rate = self._read_sample_rate()

        # Initialize the underlying MOSS-TTSD processor
        self.moss_processor = MossTTSDProcessor.from_pretrained(
            self.model_path, audio_tokenizer_path=self.xy_tokenizer_path
        )

        # Setup multimodal tokens (MOSS-TTSD uses audio tokens)
        # MOSS-TTSD doesn't have special tokens like image/video models
        # It directly processes text and audio together
        self.mm_tokens = MultimodalSpecialTokens().build(_processor)

    def _read_sample_rate(self) -> int:
        """Read output sample rate from XY tokenizer config."""
        config_path = os.path.join(self.xy_tokenizer_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
            # Use output_sample_rate for the generated audio
            input_sample_rate = config.get("input_sample_rate", 16000)
            output_sample_rate = config.get("output_sample_rate", 24000)
            return input_sample_rate, output_sample_rate

    def preprocess(
        self,
        text: str,
        system_prompt: str = "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text.",
        prompt_text: Optional[str] = None,
        prompt_audio: Optional[Union[str, bytes, np.ndarray]] = None,
        prompt_audio_speaker1: Optional[Union[str, bytes, np.ndarray]] = None,
        prompt_text_speaker1: Optional[str] = None,
        prompt_audio_speaker2: Optional[Union[str, bytes, np.ndarray]] = None,
        prompt_text_speaker2: Optional[str] = None,
    ) -> List[List[int]]:
        """Preprocess text and audio inputs to token IDs.

        Args:
            text: Text to synthesize
            system_prompt: System instruction for the model
            prompt_text: Optional text for voice cloning
            prompt_audio: Optional audio for voice cloning
            prompt_audio_speaker1: Optional audio for speaker 1 (multi-speaker)
            prompt_text_speaker1: Optional text for speaker 1
            prompt_audio_speaker2: Optional audio for speaker 2 (multi-speaker)
            prompt_text_speaker2: Optional text for speaker 2

        Returns:
            Token IDs ready for generation (2D list: [time, channels])
        """
        # Build input data dictionary
        data = {
            "text": text,
            "system_prompt": system_prompt,
        }

        # Add voice cloning data if provided
        if prompt_text:
            data["prompt_text"] = prompt_text
        if prompt_audio is not None:
            data["prompt_audio"] = self._process_audio_input(prompt_audio)

        # Add multi-speaker data if provided
        if prompt_audio_speaker1 is not None:
            data["prompt_audio_speaker1"] = self._process_audio_input(
                prompt_audio_speaker1
            )
        if prompt_text_speaker1:
            data["prompt_text_speaker1"] = prompt_text_speaker1
        if prompt_audio_speaker2 is not None:
            data["prompt_audio_speaker2"] = self._process_audio_input(
                prompt_audio_speaker2
            )
        if prompt_text_speaker2:
            data["prompt_text_speaker2"] = prompt_text_speaker2

        # Process through MOSS-TTSD processor
        inputs = self.moss_processor([data])

        # Convert to list format for JSON serialization
        # Shape is [batch, time, channels], we take first batch
        if isinstance(inputs["input_ids"], torch.Tensor):
            input_ids_list = inputs["input_ids"][0].tolist()  # Remove batch dimension
        else:
            input_ids_list = inputs["input_ids"][0]

        return input_ids_list

    def _process_audio_input(self, audio: Union[str, bytes, np.ndarray]) -> str:
        """Process audio input from various formats.

        Args:
            audio: Audio as file path, base64 string, bytes, or numpy array

        Returns:
            Path to temporary audio file
        """
        if isinstance(audio, str):
            # Check if it's a base64 string
            if not os.path.exists(audio):
                try:
                    # Try to decode as base64
                    audio_bytes = base64.b64decode(audio)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        f.write(audio_bytes)
                        return f.name
                except Exception:
                    # Assume it's a file path
                    pass
            return audio
        elif isinstance(audio, bytes):
            # Save bytes to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio)
                return f.name
        elif isinstance(audio, np.ndarray):
            # Convert numpy array to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                # Assume audio is float32 in [-1, 1] range
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                # Ensure audio is 2D (channels, samples)
                if audio.ndim == 1:
                    audio = audio[np.newaxis, :]
                torchaudio.save(
                    f.name, torch.from_numpy(audio), self.output_sample_rate
                )
                return f.name
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio)}")

    def postprocess(self, token_ids: List[List[int]]) -> Tuple[str, bytes]:
        """Postprocess generated tokens to audio bytes.

        Args:
            token_ids: Generated token IDs (2D list: [time, channels])

        Returns:
            Tuple of (decoded_text, audio_bytes)
        """
        # Convert list to tensor with batch dimension
        if isinstance(token_ids, list):
            # Add batch dimension if needed
            token_tensor = torch.tensor([token_ids])
        else:
            token_tensor = token_ids

        # Ensure correct shape [batch, time, channels]
        if token_tensor.ndim == 2:
            # Add batch dimension
            token_tensor = token_tensor.unsqueeze(0)

        # Ensure channels dimension is correct
        if token_tensor.shape[-1] != self.max_channels:
            B, T, C = (
                token_tensor.shape[0],
                token_tensor.shape[1],
                token_tensor.shape[2],
            )
            if C < self.max_channels:
                # Pad with audio pad tokens (1024 is typical pad token)
                padding = torch.full(
                    (B, T, self.max_channels - C), 1024, dtype=token_tensor.dtype
                )
                token_tensor = torch.cat([token_tensor, padding], dim=-1)
            else:
                # Truncate
                token_tensor = token_tensor[:, :, : self.max_channels]

        # Decode tokens to text and audio
        try:
            text_list, audio_list = self.moss_processor.batch_decode(token_tensor)
        except Exception as e:
            # NOTE(gy): sanitize channel-0 indices and decode manually to avoid out-of-range errors
            logger.warning(
                f"batch_decode failed ({e}); falling back to safe decode path."
            )
            try:
                # Shift back to per-timestep layout and un-offset channel-0
                normal = self.moss_processor.shifting_outputs(
                    token_tensor,
                    self.moss_processor.speech_token_range,
                    self.max_channels,
                )
                # Clamp channel-0 codes into valid range [0, 1023]
                normal[..., 0] = torch.clamp(normal[..., 0], min=0, max=1023)

                # Find valid audio spans where all non-text channels are present
                spans = self.moss_processor._find_max_valid_positions(
                    normal, self.moss_processor.audio_pad_token_id
                )

                # Decode each fragment independently using exact code lengths to avoid tail noise
                decode_audio = []
                for seq_frags in spans:
                    if len(seq_frags):
                        seq_audio = []
                        for f in seq_frags:
                            # f: [time, channels] -> [channels, batch=1, time]
                            codes = f.permute(1, 0).unsqueeze(1).contiguous()
                            code_len = torch.tensor(
                                [f.shape[0]], dtype=torch.long, device=f.device
                            )
                            out = self.moss_processor.audio_tokenizer._decode(
                                codes, codes_lengths=code_len
                            )
                            seq_audio.append(out["audio_values"][0])  # (1, T)
                        decode_audio.append(seq_audio)
                    else:
                        decode_audio.append([])

                # Decode text from channel-0 directly (keep original behavior)
                text_list = self.moss_processor.tokenizer.batch_decode(
                    token_tensor[:, :, 0]
                )
                audio_list = decode_audio
            except Exception as ee:
                logger.error(f"Safe decode path also failed: {ee}")
                raise

        # Get first result
        decoded_text = text_list[0] if text_list else ""

        # Convert audio to byte stream
        audio_bytes = b""
        if audio_list and audio_list[0]:
            # Concatenate all audio fragments
            audio_fragments = audio_list[0]
            if audio_fragments:
                # Combine all fragments
                combined_audio = torch.cat(
                    [frag.detach().cpu() for frag in audio_fragments], dim=-1
                )

                # Convert to bytes (WAV format)
                buffer = io.BytesIO()
                torchaudio.save(
                    buffer, combined_audio, self.output_sample_rate, format="wav"
                )
                audio_bytes = buffer.getvalue()

        return decoded_text, audio_bytes

    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Process multimodal data asynchronously.

        This is required by the base class but MOSS-TTSD handles
        text-to-speech differently from typical multimodal models.

        Args:
            image_data: Not used for MOSS-TTSD
            audio_data: Reference audio for voice cloning
            input_text: Text to synthesize
            request_obj: Request object with additional parameters

        Returns:
            Dictionary with processed data or None
        """
        # MOSS-TTSD processes text and generates audio
        # It doesn't follow the typical multimodal pattern of processing inputs
        # This method is mainly for compatibility with the base class

        # Extract parameters from request if available
        system_prompt = getattr(
            request_obj,
            "system_prompt",
            "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text.",
        )
        prompt_text = getattr(request_obj, "prompt_text", None)
        prompt_audio = audio_data[0] if audio_data else None

        # Preprocess to get token IDs
        input_ids = self.preprocess(
            text=input_text,
            system_prompt=system_prompt,
            prompt_text=prompt_text,
            prompt_audio=prompt_audio,
        )

        return {
            "input_ids": input_ids,
            "mm_items": [],  # MOSS-TTSD doesn't use mm_items in the same way
        }

    def process_batch(self, batch_data: List[Dict[str, Any]]) -> List[List[int]]:
        """Process a batch of requests.

        Args:
            batch_data: List of dictionaries with text and optional audio data

        Returns:
            List of token IDs for each request
        """
        results = []
        for data in batch_data:
            token_ids = self.preprocess(**data)
            results.append(token_ids)
        return results

    def postprocess_batch(
        self, batch_token_ids: List[List[List[int]]]
    ) -> List[Tuple[str, bytes]]:
        """Postprocess a batch of generated tokens.

        Args:
            batch_token_ids: List of generated token IDs

        Returns:
            List of (decoded_text, audio_bytes) tuples
        """
        results = []
        for token_ids in batch_token_ids:
            text, audio = self.postprocess(token_ids)
            results.append((text, audio))
        return results
