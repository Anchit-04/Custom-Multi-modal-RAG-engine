import numpy as np
from typing import List, Dict, Any, Optional

from indexing.vector_store import VectorStore


class Retriever:
    """
    Coarse-to-fine retrieval over the hierarchical index.

    Strategy:
      1. Search meso index (coarse) → find the most relevant time windows.
      2. For each meso hit, search micro index (fine) restricted to
         that video + time window.
      3. Merge and deduplicate micro hits.
      4. Return top-k ranked results with metadata.

    For short/precise queries (e.g. a single object or sound), callers
    can use micro_only=True to skip the meso stage.
    """

    def __init__(self, store: VectorStore, cfg):
        self.store = store
        self.cfg = cfg

    # ──────────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_vec: np.ndarray,          # (D,) float32
        video_id: Optional[str] = None, # restrict to one video if set
        micro_only: bool = False,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns a list of result dicts, each with:
            score, video_id, start_sec, end_sec,
            chunk_id, transcript, is_overlap
        Sorted by score descending.
        """
        top_k = top_k or self.cfg.rerank_top_k

        if micro_only:
            return self._micro_search(
                query_vec,
                filters={"video_id": video_id} if video_id else None,
                top_k=top_k,
            )

        return self._coarse_to_fine(query_vec, video_id, top_k)

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────

    def _coarse_to_fine(
        self,
        query_vec: np.ndarray,
        video_id: Optional[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        # Step 1 — coarse search over meso index
        meso_filters = {"video_id": video_id} if video_id else None
        meso_hits = self.store.search(
            collection=self.cfg.meso_collection,
            query_vec=query_vec,
            top_k=self.cfg.meso_top_k,
            filters=meso_filters,
        )

        if not meso_hits:
            # Fall back to flat micro search
            return self._micro_search(query_vec, meso_filters, top_k)

        # Step 2 — fine search restricted to each meso window
        micro_results: List[Dict] = []
        seen_chunk_ids = set()

        for meso_hit in meso_hits:
            payload = meso_hit["payload"]
            vid = payload.get("video_id", video_id)
            window_filters = {
                "video_id": vid,
                "start_gte": max(0.0, payload["start_sec"] - 5.0),
                "end_lte": payload["end_sec"] + 5.0,
            }
            micro_hits = self.store.search(
                collection=self.cfg.micro_collection,
                query_vec=query_vec,
                top_k=self.cfg.micro_top_k,
                filters=window_filters,
            )
            for hit in micro_hits:
                cid = hit["payload"].get("chunk_id", "")
                if cid not in seen_chunk_ids:
                    seen_chunk_ids.add(cid)
                    micro_results.append(hit)

        # Sort merged results and return top_k
        micro_results.sort(key=lambda x: x["score"], reverse=True)
        return self._format(micro_results[:top_k])

    def _micro_search(
        self,
        query_vec: np.ndarray,
        filters: Optional[Dict],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        hits = self.store.search(
            collection=self.cfg.micro_collection,
            query_vec=query_vec,
            top_k=top_k,
            filters=filters,
        )
        return self._format(hits[:top_k])

    def _format(self, hits: List[Dict]) -> List[Dict[str, Any]]:
        results = []
        for h in hits:
            p = h["payload"]
            results.append(
                {
                    "score": round(float(h["score"]), 4),
                    "video_id": p.get("video_id", ""),
                    "start_sec": p.get("start_sec", 0.0),
                    "end_sec": p.get("end_sec", 0.0),
                    "chunk_id": p.get("chunk_id", ""),
                    "transcript": p.get("transcript", ""),
                    "is_overlap": p.get("is_overlap", False),
                }
            )
        return results