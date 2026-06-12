import uuid
import numpy as np
from typing import Dict, List, Optional, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    QuantizationConfig,
    Range,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)


class VectorStore:
    """
    Thin wrapper around Qdrant for dual-vector (visual + audio) storage
    with INT8 scalar quantization (~4× RAM reduction).

    Each point stores:
      vector.visual  — (512,) float32 (quantised to int8 at rest)
      vector.audio   — (512,) float32 (quantised to int8 at rest)
      payload        — arbitrary metadata dict (video_id, start_sec, …)
    """

    def __init__(self, cfg):
        self.client = QdrantClient(path=cfg.qdrant_path)
        self.dim = cfg.shared_dim
        self._ensure_collection(cfg.micro_collection)
        self._ensure_collection(cfg.meso_collection)

    # ──────────────────────────────────────────────────────────────────────
    # Collection management
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_collection(self, name: str):
        if self.client.collection_exists(name):
            return
        self.client.create_collection(
            collection_name=name,
            vectors_config={
                "visual": VectorParams(size=self.dim, distance=Distance.COSINE),
                "audio": VectorParams(size=self.dim, distance=Distance.COSINE),
            },
            quantization_config=QuantizationConfig(
                scalar=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    )
                )
            ),
        )

    def delete_collection(self, name: str):
        if self.client.collection_exists(name):
            self.client.delete_collection(name)

    # ──────────────────────────────────────────────────────────────────────
    # Write
    # ──────────────────────────────────────────────────────────────────────

    def upsert_chunk(
        self,
        collection: str,
        chunk_id: str,
        visual_vec: np.ndarray,   # (D,) float32
        audio_vec: np.ndarray,    # (D,) float32
        payload: Dict[str, Any],
    ):
        self.client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "visual": visual_vec.tolist(),
                        "audio": audio_vec.tolist(),
                    },
                    payload={**payload, "chunk_id": chunk_id},
                )
            ],
        )

    def upsert_batch(
        self,
        collection: str,
        items: List[Dict[str, Any]],
    ):
        """
        items: list of dicts each with keys:
            chunk_id, visual_vec, audio_vec, payload
        """
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "visual": item["visual_vec"].tolist(),
                    "audio": item["audio_vec"].tolist(),
                },
                payload={**item["payload"], "chunk_id": item["chunk_id"]},
            )
            for item in items
        ]
        self.client.upsert(collection_name=collection, points=points)

    # ──────────────────────────────────────────────────────────────────────
    # Read
    # ──────────────────────────────────────────────────────────────────────

    def search(
        self,
        collection: str,
        query_vec: np.ndarray,            # (D,) float32
        top_k: int = 20,
        filters: Optional[Dict] = None,
        vis_weight: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """
        Search both visual and audio indexes, then merge scores.
        Returns list of payload dicts with an added 'score' key.
        """
        q_filter = self._build_filter(filters)

        vis_hits = self.client.search(
            collection_name=collection,
            query_vector=("visual", query_vec.tolist()),
            limit=top_k,
            query_filter=q_filter,
            with_payload=True,
        )
        aud_hits = self.client.search(
            collection_name=collection,
            query_vector=("audio", query_vec.tolist()),
            limit=top_k,
            query_filter=q_filter,
            with_payload=True,
        )
        return self._merge_hits(vis_hits, aud_hits, vis_weight)

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _merge_hits(
        self,
        vis_hits,
        aud_hits,
        vis_weight: float = 0.6,
    ) -> List[Dict[str, Any]]:
        aud_weight = 1.0 - vis_weight
        scores: Dict[str, Dict] = {}

        for h in vis_hits:
            scores[h.id] = {
                "score": h.score * vis_weight,
                "payload": h.payload,
            }
        for h in aud_hits:
            if h.id in scores:
                scores[h.id]["score"] += h.score * aud_weight
            else:
                scores[h.id] = {
                    "score": h.score * aud_weight,
                    "payload": h.payload,
                }

        return sorted(scores.values(), key=lambda x: x["score"], reverse=True)

    def _build_filter(self, filters: Optional[Dict]) -> Optional[Filter]:
        if not filters:
            return None
        conditions = []
        if "video_id" in filters:
            conditions.append(
                FieldCondition(
                    key="video_id",
                    match=MatchValue(value=filters["video_id"]),
                )
            )
        if "start_gte" in filters:
            conditions.append(
                FieldCondition(
                    key="start_sec",
                    range=Range(gte=filters["start_gte"]),
                )
            )
        if "end_lte" in filters:
            conditions.append(
                FieldCondition(
                    key="end_sec",
                    range=Range(lte=filters["end_lte"]),
                )
            )
        if not conditions:
            return None
        return Filter(must=conditions)