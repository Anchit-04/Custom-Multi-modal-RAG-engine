import gc
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from transformers import AutoModel, ASTModel

from encoding.optical_flow import OpticalFlowGate, _sample_frames
from encoding.temporal_aggregator import TemporalAggregator


class UnifiedMultimodalEncoder(nn.Module):
    """
    Unified encoder producing 512-d embeddings for:
      - Visual chunks  (VideoMAE-Small → MLP → BatchNorm → L2)
      - Audio chunks   (AST-finetuned  → MLP → BatchNorm → L2)
      - Text queries   (CLIP ViT-B/32  → Linear          → L2)

    Backbones are frozen; only projection heads, BN layers,
    temporal aggregator, and optical flow gate are trainable.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ── Visual backbone (VideoMAE-Small, 384-d output) ──────────────
        self.visual_backbone = AutoModel.from_pretrained(
            "MCG-NJU/videomae-small"
        )
        _freeze(self.visual_backbone)

        # ── Audio backbone (AST AudioSet, 768-d CLS output) ─────────────
        self.audio_backbone = ASTModel.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        _freeze(self.audio_backbone)

        # ── Text backbone (CLIP ViT-B/32, 512-d output) ─────────────────
        clip_model, _, self.clip_preprocess = (
            open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        )
        self.text_backbone = clip_model
        _freeze(self.text_backbone)

        # ── Projection heads ─────────────────────────────────────────────
        vis_in = 384
        aud_in = 768
        txt_in = 512
        d = cfg.shared_dim

        self.vis_proj = _mlp(vis_in, d)
        self.aud_proj = _mlp(aud_in, d)
        self.text_proj = nn.Linear(txt_in, d)   # text already well-aligned

        # ── Normalisation (BatchNorm1d for contrastive training) ─────────
        self.bn_vis = nn.BatchNorm1d(d)
        self.bn_aud = nn.BatchNorm1d(d)

        # ── Motion gate (CPU — zero VRAM overhead) ───────────────────────
        self.flow_gate = OpticalFlowGate(d, num_frames=cfg.frames_per_chunk)

        # ── Temporal aggregator ──────────────────────────────────────────
        self.temporal_agg = TemporalAggregator(
            shared_dim=d,
            num_frames=cfg.frames_per_chunk,
            num_layers=cfg.temporal_layers,
            num_heads=cfg.temporal_heads,
        )

        # ── Learnable temperature (CLIP-style contrastive) ───────────────
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1.0 / 0.07))
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public encode methods
    # ──────────────────────────────────────────────────────────────────────

    def encode_vision(
        self,
        pixel_values: torch.Tensor,                  # (B, T, C, H, W)
        raw_frames_batch: Optional[List[List[np.ndarray]]] = None,
        chunk_positions: Optional[torch.Tensor] = None,  # (B,) int64
    ) -> torch.Tensor:
        """
        Returns (B, shared_dim) L2-normalised visual embedding.

        pixel_values should be preprocessed by the VideoMAE processor
        (normalised, resized to 224×224).
        """
        B, T, C, H, W = pixel_values.shape

        with torch.no_grad():
            out = self.visual_backbone(pixel_values=pixel_values)
            # last_hidden_state: (B, num_patches, 384)
            x_v = out.last_hidden_state.mean(dim=1).float()  # (B, 384)

        # Project → BN (BN needs training mode to compute running stats)
        z_v = self.bn_vis(self.vis_proj(x_v))  # (B, d)

        # Per-frame embeddings for temporal aggregator
        # Re-use the pooled embedding tiled T times as a lightweight approximation
        # (a full per-frame VideoMAE pass would cost T× the VRAM)
        frame_embeds = z_v.unsqueeze(1).expand(-1, T, -1)  # (B, T, d)
        z_v = self.temporal_agg(frame_embeds, chunk_positions)  # (B, d) normalised

        # Optical flow motion gate (CPU)
        if raw_frames_batch is not None:
            z_v = self.flow_gate(z_v, raw_frames_batch)

        return z_v   # already L2-normalised by temporal_agg / flow_gate

    def encode_audio(
        self,
        input_values: torch.Tensor,  # (B, mel_bins, time_frames) — AST input
    ) -> torch.Tensor:
        """Returns (B, shared_dim) L2-normalised audio embedding."""
        with torch.no_grad():
            out = self.audio_backbone(input_values=input_values)
            x_a = out.last_hidden_state[:, 0, :].float()  # CLS token (B, 768)

        z_a = self.bn_aud(self.aud_proj(x_a))
        return F.normalize(z_a, p=2, dim=-1)

    def encode_text(
        self,
        text_tokens: torch.Tensor,  # (B, seq_len) tokenised by open_clip
    ) -> torch.Tensor:
        """Returns (B, shared_dim) L2-normalised text embedding."""
        with torch.no_grad():
            x_t = self.text_backbone.encode_text(text_tokens).float()  # (B, 512)

        z_t = self.text_proj(x_t)
        return F.normalize(z_t, p=2, dim=-1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_values: torch.Tensor,
        raw_frames_batch: Optional[List[List[np.ndarray]]] = None,
        chunk_positions: Optional[torch.Tensor] = None,
    ):
        """Simultaneous forward for contrastive training / joint inference."""
        z_v = self.encode_vision(pixel_values, raw_frames_batch, chunk_positions)
        z_a = self.encode_audio(input_values)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        return z_v, z_a, logit_scale


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def _mlp(in_dim: int, out_dim: int, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim * 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(out_dim * 2, out_dim),
    )