import numpy as np
from typing import List
from ingestion.extractor import RawVideo
from chunking.duration_enforcer import VideoChunk, _time_to_frame


def inject_overlaps(
    chunks: List[VideoChunk],
    raw: RawVideo,
    cfg,
) -> List[VideoChunk]:
    """
    For every boundary between consecutive chunks, create an additional
    overlap chunk of cfg.overlap_sec centred on the boundary.
    These overlap chunks are tagged with '_overlap' in their chunk_id and
    will be indexed alongside regular chunks.

    This ensures that queries spanning a boundary region are retrievable.
    """
    if len(chunks) < 2:
        return chunks

    audio_sr = cfg.audio_sr
    overlap_half = cfg.overlap_sec / 2.0
    n_frames = len(raw.frames)

    overlap_chunks: List[VideoChunk] = []

    for i in range(len(chunks) - 1):
        boundary_sec = chunks[i].end_sec

        t_start = max(0.0, boundary_sec - overlap_half)
        t_end = min(raw.duration, boundary_sec + overlap_half)

        # Skip if the resulting window is too short to be useful
        if t_end - t_start < 1.0:
            continue

        f_start = _time_to_frame(t_start, raw.frame_timestamps)
        f_end = min(_time_to_frame(t_end, raw.frame_timestamps), n_frames)

        chunk_frames = raw.frames[f_start:f_end]
        if len(chunk_frames) == 0:
            continue

        a_start = int(t_start * audio_sr)
        a_end = int(t_end * audio_sr)
        audio_seg = raw.audio[a_start:a_end]

        overlap_chunks.append(
            VideoChunk(
                chunk_id=f"overlap_{i:05d}",
                start_sec=t_start,
                end_sec=t_end,
                start_frame=f_start,
                end_frame=f_end,
                raw_frames=chunk_frames,
                audio_segment=audio_seg,
                transcript_text="",  # overlap chunks don't carry transcript
            )
        )

    # Return interleaved: regular chunks + overlap chunks sorted by time
    all_chunks = chunks + overlap_chunks
    all_chunks.sort(key=lambda c: c.start_sec)
    return all_chunks