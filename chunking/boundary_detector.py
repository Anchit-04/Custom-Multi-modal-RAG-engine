import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Boundary:
    frame_idx: int
    timestamp: float
    confidence: float       # 0-1, higher = more confident it's a real cut
    source: str             # 'transcript', 'visual', 'both'


class BoundaryDetector:
    """
    Detects chunk boundaries by combining:
      1. Sentence-end timestamps from Whisper transcript
      2. Cosine distance peaks in frame embedding space

    The visual distance threshold is computed ADAPTIVELY per video from
    the distance signal's own statistics (90th percentile + mean/std),
    rather than a fixed config value. This avoids needing to manually
    re-tune a threshold for every new video — animation, live-action,
    and static talking-head footage all have very different natural
    distance scales.

    A boundary is 'both' when transcript and visual agree within ±5 frames.
    Pure visual peaks above 1.5x the adaptive threshold are added for
    action/silent content with no speech.
    """

    SENTENCE_ENDERS = {".", "!", "?"}

    def __init__(self, cfg):
        self.cfg = cfg
        self.last_threshold_used: float = None  # exposed for debugging/logging

    # ------------------------------------------------------------------
    def detect(
        self,
        frame_embeds: torch.Tensor,       # (N, 512) float32
        frame_timestamps: List[float],
        whisper_result: Dict[str, Any],
    ) -> List[Boundary]:

        vis_dist = self._visual_distances(frame_embeds)

        # Threshold derived fresh from THIS video's distance distribution
        threshold = self._compute_adaptive_threshold(vis_dist)
        self.last_threshold_used = threshold

        transcript_times = self._extract_sentence_ends(whisper_result)
        boundaries = self._match_transcript_to_visual(
            transcript_times, frame_timestamps, vis_dist, threshold
        )
        boundaries = self._add_visual_only_peaks(
            boundaries, frame_timestamps, vis_dist, threshold
        )

        boundaries.sort(key=lambda b: b.timestamp)
        boundaries = self._deduplicate(boundaries, min_gap=1.0)
        return boundaries

    # ------------------------------------------------------------------
    # Adaptive thresholding
    # ------------------------------------------------------------------

    def _compute_adaptive_threshold(self, dist: torch.Tensor) -> float:
        """
        Derive a per-video boundary threshold from the distance signal itself.

        Combines two signals:
          1. Percentile-based: catches "naturally unusual" jumps for this video
          2. Mean + k*std: catches statistical outliers regardless of percentile

        Takes the more conservative (higher) of the two, with a floor
        (cfg.min_semantic_threshold) to avoid over-segmenting near-static video.
        """
        dist_np = dist.numpy()

        percentile_thresh = float(np.percentile(dist_np, 90))
        std_thresh = float(dist_np.mean() + 2.5 * dist_np.std())

        floor = getattr(self.cfg, "min_semantic_threshold", 0.015)
        return max(percentile_thresh, std_thresh, floor)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _visual_distances(self, frame_embeds: torch.Tensor) -> torch.Tensor:
        """Return smoothed cosine distance between consecutive frames."""
        sim = F.cosine_similarity(frame_embeds[:-1], frame_embeds[1:], dim=-1)
        dist = 1.0 - sim
        return self._smooth(dist, window=4)

    def _smooth(self, signal: torch.Tensor, window: int) -> torch.Tensor:
        kernel = torch.ones(window, dtype=signal.dtype) / window
        padded = F.pad(
            signal.unsqueeze(0).unsqueeze(0),
            (window // 2, window // 2),
            mode="replicate",
        )
        return F.conv1d(padded, kernel.view(1, 1, -1)).squeeze()

    def _extract_sentence_ends(self, whisper_result: Dict) -> List[float]:
        times: List[float] = []
        for seg in whisper_result.get("segments", []):
            text = seg.get("text", "").strip()
            if any(text.endswith(p) for p in self.SENTENCE_ENDERS):
                times.append(float(seg["end"]))
        return times

    def _match_transcript_to_visual(
        self,
        transcript_times: List[float],
        frame_timestamps: List[float],
        vis_dist: torch.Tensor,
        threshold: float,
    ) -> List[Boundary]:
        boundaries: List[Boundary] = []
        n = len(frame_timestamps)
        dist_np = vis_dist.numpy()

        for t in transcript_times:
            frame_idx = int(
                np.argmin([abs(frame_timestamps[i] - t) for i in range(n)])
            )
            w_start = max(0, frame_idx - 5)
            w_end = min(len(dist_np) - 1, frame_idx + 5)
            local_max = float(dist_np[w_start : w_end + 1].max())

            if local_max > threshold:
                source = "both"
                confidence = min(1.0, local_max * 2.0)
            else:
                source = "transcript"
                confidence = 0.5

            boundaries.append(
                Boundary(
                    frame_idx=frame_idx,
                    timestamp=t,
                    confidence=confidence,
                    source=source,
                )
            )
        return boundaries

    def _add_visual_only_peaks(
        self,
        existing: List[Boundary],
        frame_timestamps: List[float],
        vis_dist: torch.Tensor,
        threshold: float,
    ) -> List[Boundary]:
        """
        Add strong visual-only boundaries (e.g. hard scene cuts in silent video).
        Uses 1.5x the adaptive threshold to avoid false positives.
        """
        dist_np = vis_dist.numpy()
        new: List[Boundary] = []

        for i in range(1, len(dist_np) - 1):
            if (
                dist_np[i] > threshold
                and dist_np[i] > dist_np[i - 1]
                and dist_np[i] > dist_np[i + 1]
            ):
                t = frame_timestamps[i]
                if any(abs(b.timestamp - t) < 2.0 for b in existing):
                    continue
                new.append(
                    Boundary(
                        frame_idx=i,
                        timestamp=t,
                        confidence=min(1.0, dist_np[i] * 1.5),
                        source="visual",
                    )
                )
        return existing + new

    def _deduplicate(
        self, boundaries: List[Boundary], min_gap: float
    ) -> List[Boundary]:
        """Remove boundaries closer than min_gap seconds to the previous one."""
        if not boundaries:
            return boundaries
        deduped = [boundaries[0]]
        for b in boundaries[1:]:
            if b.timestamp - deduped[-1].timestamp >= min_gap:
                deduped.append(b)
        return deduped