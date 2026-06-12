import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class OpticalFlowGate(nn.Module):
    """
    Computes Farneback optical flow on CPU from raw frame pairs,
    derives a compact motion descriptor, projects it to shared_dim,
    then soft-gates it with the appearance embedding.

    Motion descriptor (per chunk):
      - mean optical flow magnitude per consecutive frame pair  (T-1 dims)
      - direction histogram (8 bins)                           (8 dims)
    Total: (T-1) + 8 dims → projected to shared_dim via a 2-layer MLP.

    The gate learns to weight appearance vs. motion adaptively —
    for static talking-head clips the gate suppresses motion almost
    entirely; for action scenes it boosts it.
    """

    def __init__(self, shared_dim: int, num_frames: int = 8):
        super().__init__()
        self.shared_dim = shared_dim
        self.num_frames = num_frames
        flow_desc_dim = (num_frames - 1) + 8

        self.flow_proj = nn.Sequential(
            nn.Linear(flow_desc_dim, 128),
            nn.GELU(),
            nn.Linear(128, shared_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(shared_dim * 2, shared_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        appearance_embed: torch.Tensor,   # (B, shared_dim) on any device
        raw_frames_batch: List[List[np.ndarray]],  # list of B lists of np frames
    ) -> torch.Tensor:
        """
        raw_frames_batch: outer list = batch items, inner list = frames per chunk.
        Frames are uint8 RGB numpy arrays (H, W, 3).
        Returns fused embedding of shape (B, shared_dim), same device as input.
        """
        device = appearance_embed.device
        flow_descs = []
        for frames in raw_frames_batch:
            desc = self._compute_flow_descriptor(frames)
            flow_descs.append(desc)

        flow_tensor = torch.tensor(
            np.stack(flow_descs), dtype=torch.float32
        ).to(device)                        # (B, flow_desc_dim)

        motion_embed = self.flow_proj(flow_tensor)  # (B, shared_dim)

        gate_input = torch.cat([appearance_embed, motion_embed], dim=-1)
        alpha = self.gate(gate_input)               # (B, shared_dim)

        fused = alpha * appearance_embed + (1.0 - alpha) * motion_embed
        return F.normalize(fused, p=2, dim=-1)

    # ------------------------------------------------------------------

    def _compute_flow_descriptor(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Given a list of RGB frames, compute per-pair optical flow and
        return a fixed-length descriptor vector.
        """
        # Sample exactly self.num_frames evenly from the chunk
        frames = _sample_frames(frames, self.num_frames)

        mag_means: List[float] = []
        all_directions: List[np.ndarray] = []

        for i in range(len(frames) - 1):
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                gray1,
                gray2,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )  # (H, W, 2)

            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            direction = np.arctan2(flow[..., 1], flow[..., 0])

            mag_means.append(float(magnitude.mean()))
            all_directions.append(direction.ravel())

        # Direction histogram (8 bins) over all frame pairs
        if all_directions:
            all_dir = np.concatenate(all_directions)
            dir_hist, _ = np.histogram(all_dir, bins=8, range=(-np.pi, np.pi))
            dir_hist = dir_hist.astype(np.float32)
            total = dir_hist.sum() + 1e-6
            dir_hist /= total
        else:
            dir_hist = np.zeros(8, dtype=np.float32)

        descriptor = np.array(mag_means, dtype=np.float32)
        # Pad if fewer pairs than expected
        if len(descriptor) < self.num_frames - 1:
            descriptor = np.pad(
                descriptor,
                (0, self.num_frames - 1 - len(descriptor)),
                mode="edge",
            )

        return np.concatenate([descriptor, dir_hist])   # (num_frames-1+8,)


# ------------------------------------------------------------------

def _sample_frames(
    frames: List[np.ndarray], n: int
) -> List[np.ndarray]:
    """Uniformly sample exactly n frames from a list."""
    if len(frames) <= n:
        return frames
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in indices]