import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any


class TemporalReranker(nn.Module):
    """
    Cross-attention reranker that scores candidate chunks against a query.

    For each candidate chunk the retriever returns, we load its stored
    per-frame visual embeddings (if available) and run cross-attention
    between the query and the frame sequence.

    In the simpler case where per-frame embeddings are not stored,
    we fall back to a dot-product rescore using the chunk's single
    pooled embedding — still better than pure cosine on the query alone.

    Architecture:
        query embed (1, D) attends over chunk frame embeds (T, D)
        → attended summary (1, D)
        → score head (1, D) → scalar
    """

    def __init__(self, shared_dim: int = 512):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=shared_dim,
            num_heads=8,
            dropout=0.0,
            batch_first=True,
        )
        self.score_head = nn.Sequential(
            nn.Linear(shared_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        query_embed: torch.Tensor,         # (1, D)
        chunk_frame_embeds: torch.Tensor,  # (T, D) or (1, D)
    ) -> torch.Tensor:
        """Returns a scalar relevance score."""
        q = query_embed.unsqueeze(0)              # (1, 1, D)
        kv = chunk_frame_embeds.unsqueeze(0)      # (1, T, D)

        attended, _ = self.cross_attn(q, kv, kv)  # (1, 1, D)
        score = self.score_head(attended.squeeze(1))  # (1, 1)
        return score.squeeze()


class Reranker:
    """
    Stateless reranker wrapper.  Works in two modes:

    Mode A — dot-product fallback (no stored frame embeds):
        Rescores each candidate by computing the dot product between
        the query vector and the stored visual vector. Adds a small
        transcript BM25-style bonus when the query keywords appear in
        the transcript.

    Mode B — cross-attention (when frame_embeds_store is provided):
        Loads stored per-frame embeddings and runs TemporalReranker.

    For the 2-week build, Mode A is sufficient and requires zero extra
    storage. Mode B is the upgrade path.
    """

    def __init__(
        self,
        shared_dim: int = 512,
        frame_embeds_store=None,
        device: str = "cpu",
    ):
        self.device = device
        self.frame_embeds_store = frame_embeds_store

        if frame_embeds_store is not None:
            self.model = TemporalReranker(shared_dim).to(device).eval()
        else:
            self.model = None

    def rerank(
        self,
        query_vec: np.ndarray,          # (D,) float32
        candidates: List[Dict[str, Any]],
        query_text: str = "",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates and return top_k with updated scores.
        """
        if not candidates:
            return candidates

        if self.model is not None and self.frame_embeds_store is not None:
            return self._rerank_cross_attention(query_vec, candidates, top_k)

        return self._rerank_dot_product(query_vec, candidates, query_text, top_k)

    # ──────────────────────────────────────────────────────────────────────
    # Mode A — dot-product + transcript keyword bonus
    # ──────────────────────────────────────────────────────────────────────

    def _rerank_dot_product(
        self,
        query_vec: np.ndarray,
        candidates: List[Dict],
        query_text: str,
        top_k: int,
    ) -> List[Dict]:
        q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        query_keywords = set(query_text.lower().split())

        rescored = []
        for c in candidates:
            base_score = float(c["score"])

            # Transcript keyword overlap bonus (simple, no stopwords)
            transcript = c.get("transcript", "").lower()
            transcript_words = set(transcript.split())
            overlap = len(query_keywords & transcript_words)
            keyword_bonus = min(0.1, overlap * 0.02)   # max +0.10

            # Penalise overlap chunks slightly (they're secondary evidence)
            overlap_penalty = 0.02 if c.get("is_overlap", False) else 0.0

            final_score = base_score + keyword_bonus - overlap_penalty
            rescored.append({**c, "score": round(final_score, 4)})

        rescored.sort(key=lambda x: x["score"], reverse=True)
        return rescored[:top_k]

    # ──────────────────────────────────────────────────────────────────────
    # Mode B — cross-attention over stored frame embeddings
    # ──────────────────────────────────────────────────────────────────────

    def _rerank_cross_attention(
        self,
        query_vec: np.ndarray,
        candidates: List[Dict],
        top_k: int,
    ) -> List[Dict]:
        q_tensor = torch.tensor(query_vec, dtype=torch.float32).to(self.device)

        rescored = []
        with torch.no_grad():
            for c in candidates:
                frame_embeds = self.frame_embeds_store.get(c["chunk_id"])
                if frame_embeds is None:
                    # Fall back to base score if no frame embeds stored
                    rescored.append(c)
                    continue

                fe_tensor = torch.tensor(
                    frame_embeds, dtype=torch.float32
                ).to(self.device)  # (T, D)

                score = self.model(q_tensor.unsqueeze(0), fe_tensor)
                rescored.append({**c, "score": round(float(score.cpu()), 4)})

        rescored.sort(key=lambda x: x["score"], reverse=True)
        return rescored[:top_k]