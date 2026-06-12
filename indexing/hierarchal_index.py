import numpy as np
import torch
from typing import List, Tuple, Dict, Any

from indexing.vector_store import VectorStore


class HierarchicalIndex:
    """
    Maintains two levels of index:
      micro  — individual chunks (3–45 s), fine-grained retrieval
      meso   — groups of cfg.meso_group_size micro chunks, coarse retrieval

    Meso embeddings are computed by mean-pooling the visual and audio
    embeddings of their constituent micro chunks.

    During query time, the retriever can choose:
      - micro-only    for short precise queries
      - coarse-to-fine: search meso → restrict micro search to that time window
    """

    def __init__(self, store: VectorStore, cfg):
        self.store = store
        self.cfg = cfg
        self.micro_col = cfg.micro_collection
        self.meso_col = cfg.meso_collection

    # ──────────────────────────────────────────────────────────────────────
    # Indexing
    # ──────────────────────────────────────────────────────────────────────

    def index_all(
        self,
        embeddings: List[Tuple[torch.Tensor, torch.Tensor, Any]],
        # each item: (z_v (1,D), z_a (1,D), VideoChunk)
        video_id: str,
    ):
        """
        Index all micro chunks, then build and index meso chunks.
        """
        micro_items = []
        for z_v, z_a, chunk in embeddings:
            vis_np = z_v.squeeze(0).numpy().astype(np.float32)
            aud_np = z_a.squeeze(0).numpy().astype(np.float32)
            payload = {
                "video_id": video_id,
                "start_sec": chunk.start_sec,
                "end_sec": chunk.end_sec,
                "transcript": chunk.transcript_text,
                "is_overlap": "overlap" in chunk.chunk_id,
            }
            micro_items.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "visual_vec": vis_np,
                    "audio_vec": aud_np,
                    "payload": payload,
                }
            )

        # Batch upsert micro
        if micro_items:
            self.store.upsert_batch(self.micro_col, micro_items)

        # Build and upsert meso
        meso_items = self._build_meso(micro_items, video_id)
        if meso_items:
            self.store.upsert_batch(self.meso_col, meso_items)

    def _build_meso(
        self,
        micro_items: List[Dict],
        video_id: str,
    ) -> List[Dict]:
        """
        Group consecutive micro chunks into meso groups.
        Exclude overlap chunks from grouping (they don't contribute to meso).
        """
        regular = [
            item for item in micro_items if not item["payload"]["is_overlap"]
        ]
        group_size = self.cfg.meso_group_size
        meso_items = []

        for i in range(0, len(regular), group_size):
            group = regular[i : i + group_size]
            if not group:
                continue

            vis_mean = np.mean(
                [g["visual_vec"] for g in group], axis=0
            ).astype(np.float32)
            aud_mean = np.mean(
                [g["audio_vec"] for g in group], axis=0
            ).astype(np.float32)

            # L2 re-normalise the mean
            vis_mean /= np.linalg.norm(vis_mean) + 1e-8
            aud_mean /= np.linalg.norm(aud_mean) + 1e-8

            start_sec = group[0]["payload"]["start_sec"]
            end_sec = group[-1]["payload"]["end_sec"]
            transcript = " ".join(
                g["payload"].get("transcript", "") for g in group
            ).strip()

            meso_items.append(
                {
                    "chunk_id": f"meso_{i // group_size:05d}",
                    "visual_vec": vis_mean,
                    "audio_vec": aud_mean,
                    "payload": {
                        "video_id": video_id,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "transcript": transcript,
                        "is_overlap": False,
                    },
                }
            )

        return meso_items