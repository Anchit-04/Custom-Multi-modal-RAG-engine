%%writefile /content/Custom-Multi-modal-RAG-engine/indexing/vector_store.py
import uuid
import numpy as np
from typing import Dict, List, Optional, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    NamedVector,
    PointStruct,
    Range,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)


class VectorStore:
    def __init__(self, cfg):
        self.client = QdrantClient(path=cfg.qdrant_path)
        self.dim = cfg.shared_dim
        self._ensure_collection(cfg.micro_collection)
        self._ensure_collection(cfg.meso_collection)

    def _ensure_collection(self, name: str):
        if self.client.collection_exists(name):
            return
        self.client.create_collection(
            collection_name=name,
            vectors_config={
                "visual": VectorParams(size=self.dim, distance=Distance.COSINE),
                "audio": VectorParams(size=self.dim, distance=Distance.COSINE),
            },
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            ),
        )

    def delete_collection(self, name: str):
        if self.client.collection_exists(name):
            self.client.delete_collection(name)

    def upsert_chunk(self, collection, chunk_id, visual_vec, audio_vec, payload):
        self.client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={"visual": visual_vec.tolist(), "audio": audio_vec.tolist()},
                    payload={**payload, "chunk_id": chunk_id},
                )
            ],
        )

    def upsert_batch(self, collection, items):
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

    def search(self, collection, query_vec, top_k=20, filters=None, vis_weight=0.6):
        q_filter = self._build_filter(filters)
        query_list = query_vec.tolist()

        vis_hits = self.client.query_points(
            collection_name=collection,
            query=query_list,
            using="visual",
            limit=top_k,
            query_filter=q_filter,
            with_payload=True,
        ).points

        aud_hits = self.client.query_points(
            collection_name=collection,
            query=query_list,
            using="audio",
            limit=top_k,
            query_filter=q_filter,
            with_payload=True,
        ).points

        return self._merge_hits(vis_hits, aud_hits, vis_weight)

    def _merge_hits(self, vis_hits, aud_hits, vis_weight=0.6):
        aud_weight = 1.0 - vis_weight
        scores = {}
        for h in vis_hits:
            scores[h.id] = {"score": h.score * vis_weight, "payload": h.payload}
        for h in aud_hits:
            if h.id in scores:
                scores[h.id]["score"] += h.score * aud_weight
            else:
                scores[h.id] = {"score": h.score * aud_weight, "payload": h.payload}
        return sorted(scores.values(), key=lambda x: x["score"], reverse=True)

    def _build_filter(self, filters):
        if not filters:
            return None
        conditions = []
        if "video_id" in filters:
            conditions.append(
                FieldCondition(key="video_id", match=MatchValue(value=filters["video_id"]))
            )
        if "start_gte" in filters:
            conditions.append(
                FieldCondition(key="start_sec", range=Range(gte=filters["start_gte"]))
            )
        if "end_lte" in filters:
            conditions.append(
                FieldCondition(key="end_sec", range=Range(lte=filters["end_lte"]))
            )
        if not conditions:
            return None
        return Filter(must=conditions)