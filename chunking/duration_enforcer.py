import numpy as np
import torch
import librosa
from dataclasses import dataclass, field
from typing import List, Optional
from ingestion.extractor import RawVideo
from chunking.boundary_detector import Boundary


@dataclass
class VideoChunk:
    chunk_id: str
    start_sec: float
    end_sec: float
    start_frame: int
    end_frame: int
    raw_frames: List[np.ndarray]       # list of (H, W, 3) uint8
    audio_segment: np.ndarray          # float32 mono 16kHz
    transcript_text: str = ""


def enforce_duration(
    raw: RawVideo,
    boundaries: List[Boundary],
    cfg,
    transcript_segments: Optional[List[dict]] = None,
) -> List[VideoChunk]:
    """
    Convert boundaries into VideoChunk objects, enforcing min/max duration.

    Rules:
    - If gap between boundaries < min_chunk_sec: merge with next boundary.
    - If gap > max_chunk_sec: force-split at max_chunk_sec intervals.
    - Chunks shorter than min_chunk_sec at the end are merged into the previous.
    """
    duration = raw.duration
    n_frames = len(raw.frames)

    # Build list of cut points in seconds (include video start and end)
    cut_times = [0.0] + [b.timestamp for b in boundaries] + [duration]
    cut_times = sorted(set(cut_times))

    # --- Merge cuts that are too close ---
    merged: List[float] = [cut_times[0]]
    for t in cut_times[1:]:
        if t - merged[-1] < cfg.min_chunk_sec:
            continue  # skip — too short, will be absorbed by next
        merged.append(t)
    if merged[-1] < duration:
        merged.append(duration)

    # --- Force-split segments that are too long ---
    final_cuts: List[float] = []
    for i in range(len(merged) - 1):
        seg_start = merged[i]
        seg_end = merged[i + 1]
        seg_dur = seg_end - seg_start
        final_cuts.append(seg_start)
        if seg_dur > cfg.max_chunk_sec:
            n_splits = int(np.ceil(seg_dur / cfg.max_chunk_sec))
            step = seg_dur / n_splits
            for k in range(1, n_splits):
                final_cuts.append(seg_start + k * step)
    final_cuts.append(duration)
    final_cuts = sorted(set(final_cuts))

    # --- Build chunks ---
    chunks: List[VideoChunk] = []
    audio_sr = cfg.audio_sr  # 16000

    for i in range(len(final_cuts) - 1):
        t_start = final_cuts[i]
        t_end = final_cuts[i + 1]

        # Frame indices (inclusive start, exclusive end)
        f_start = _time_to_frame(t_start, raw.frame_timestamps)
        f_end = _time_to_frame(t_end, raw.frame_timestamps)
        f_end = min(f_end, n_frames)

        chunk_frames = raw.frames[f_start:f_end]
        if len(chunk_frames) == 0:
            continue

        # Audio slice
        a_start = int(t_start * audio_sr)
        a_end = int(t_end * audio_sr)
        audio_seg = raw.audio[a_start:a_end]

        # Transcript text for this window (optional)
        text = _get_transcript_text(t_start, t_end, transcript_segments or [])

        chunk = VideoChunk(
            chunk_id=f"chunk_{i:05d}",
            start_sec=t_start,
            end_sec=t_end,
            start_frame=f_start,
            end_frame=f_end,
            raw_frames=chunk_frames,
            audio_segment=audio_seg,
            transcript_text=text,
        )
        chunks.append(chunk)

    return chunks


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _time_to_frame(t: float, timestamps: List[float]) -> int:
    """Return index of the frame closest to time t."""
    if t <= timestamps[0]:
        return 0
    if t >= timestamps[-1]:
        return len(timestamps) - 1
    diffs = [abs(ts - t) for ts in timestamps]
    return int(np.argmin(diffs))


def _get_transcript_text(
    t_start: float,
    t_end: float,
    segments: List[dict],
) -> str:
    words = []
    for seg in segments:
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", 0.0)
        if seg_start >= t_end or seg_end <= t_start:
            continue
        words.append(seg.get("text", "").strip())
    return " ".join(words)