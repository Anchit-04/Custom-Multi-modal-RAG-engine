"""
Converts raw frames (numpy uint8) and raw audio (float32 numpy)
into the tensor formats expected by VideoMAE and AST.

This lives separately from the encoder so the pipeline can preprocess
on CPU while the GPU encodes the previous batch.
"""

import numpy as np
import torch
from typing import List
from transformers import AutoFeatureExtractor, ASTFeatureExtractor
from PIL import Image


# ── Lazy singletons so we only instantiate once per process ──────────────────

_videomae_extractor = None
_ast_extractor = None


def get_videomae_extractor():
    global _videomae_extractor
    if _videomae_extractor is None:
        _videomae_extractor = AutoFeatureExtractor.from_pretrained(
            "MCG-NJU/videomae-small"
        )
    return _videomae_extractor


def get_ast_extractor():
    global _ast_extractor
    if _ast_extractor is None:
        _ast_extractor = ASTFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
    return _ast_extractor


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess_frames(
    raw_frames: List[np.ndarray],
    num_frames: int = 8,
) -> torch.Tensor:
    """
    Sample `num_frames` from raw_frames uniformly, resize to 224×224,
    and return a (1, T, C, H, W) float32 tensor ready for VideoMAE.

    raw_frames: list of (H, W, 3) uint8 RGB numpy arrays.
    """
    extractor = get_videomae_extractor()

    sampled = _sample_evenly(raw_frames, num_frames)

    # Convert to PIL for the HF extractor
    pil_frames = [Image.fromarray(f) for f in sampled]

    # The VideoMAE feature extractor returns pixel_values: (1, T, C, H, W)
    inputs = extractor(images=pil_frames, return_tensors="pt")
    return inputs["pixel_values"]  # (1, T, C, H, W)


def preprocess_audio(
    audio_segment: np.ndarray,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Convert a mono float32 audio array into the log-mel spectrogram
    expected by AST and return a (1, mel_bins, time_frames) tensor.

    audio_segment: (samples,) float32.
    """
    extractor = get_ast_extractor()

    inputs = extractor(
        audio_segment,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    # AST extractor returns input_values: (1, time_frames, mel_bins)
    # The ASTModel expects (batch, time, mel) so we keep it as-is
    return inputs["input_values"]  # (1, time_frames, mel_bins)


# ── Helper ────────────────────────────────────────────────────────────────────

def _sample_evenly(frames: List[np.ndarray], n: int) -> List[np.ndarray]:
    if len(frames) <= n:
        # Pad by repeating last frame
        return frames + [frames[-1]] * (n - len(frames))
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in indices]