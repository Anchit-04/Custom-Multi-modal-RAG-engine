"""
Main pipeline.  Two public entry points:

    pipeline = Pipeline()
    pipeline.index_video("path/to/video.mp4", video_id="my_vid")
    results  = pipeline.query("man shooting another man", video_id="my_vid")
"""

import gc
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch

from config import cfg
from ingestion.extractor import extract
from ingestion.whisper_runner import transcribe
from chunking.boundary_detector import BoundaryDetector
from chunking.duration_enforcer import enforce_duration
from chunking.overlap_injector import inject_overlaps
from encoding.encoder import UnifiedMultimodalEncoder
from encoding.preprocessor import preprocess_frames, preprocess_audio
from indexing.vector_store import VectorStore
from indexing.hierarchal_index import HierarchicalIndex
from retrieval.query_encoder import QueryEncoder
from retrieval.retriever import Retriever
from retrieval.reranker import Reranker


class Pipeline:
    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.store = VectorStore(cfg)
        self.index = HierarchicalIndex(self.store, cfg)
        self.retriever = Retriever(self.store, cfg)
        self.reranker = Reranker(shared_dim=cfg.shared_dim, device=self.device)

        # Query encoder is kept alive (lightweight CLIP text only)
        self.query_encoder = QueryEncoder(encoder=None, device=self.device)

        print(f"[Pipeline] device={self.device}")

    # ──────────────────────────────────────────────────────────────────────
    # Indexing
    # ──────────────────────────────────────────────────────────────────────

    def index_video(self, video_path: str, video_id: str):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"[1/4] Extracting  {video_path}")
        raw = extract(video_path, target_fps=cfg.frame_fps)
        print(f"      {len(raw.frames)} frames · {raw.duration:.1f}s · "
              f"{len(raw.audio)/cfg.audio_sr:.1f}s audio")

        # ── Phase 2a: Transcribe (loads Whisper, frees after) ────────────
        print(f"[2/4] Transcribing  (whisper-{cfg.whisper_model})")
        transcript = transcribe(video_path, cfg.whisper_model)
        print(f"      {len(transcript.get('words', []))} words detected")

        # ── Phase 2b: Get rough frame embeddings for boundary detection ──
        print("      Computing rough frame embeddings for boundary detection")
        rough_embeds = self._rough_encode_frames(raw.frames)

        # ── Phase 2c: Detect boundaries ──────────────────────────────────
        detector = BoundaryDetector(cfg)
        boundaries = detector.detect(
            rough_embeds,
            raw.frame_timestamps,
            transcript,
        )
        print(f"      {len(boundaries)} boundaries detected")

        # ── Phase 2d: Enforce duration + inject overlaps ─────────────────
        chunks = enforce_duration(
            raw, boundaries, cfg,
            transcript_segments=transcript.get("segments", []),
        )
        chunks = inject_overlaps(chunks, raw, cfg)
        print(f"      {len(chunks)} chunks after overlap injection")

        # ── Phase 3: Encode (GPU) ─────────────────────────────────────────
        print(f"[3/4] Encoding  (device={self.device})")
        encoder = UnifiedMultimodalEncoder(cfg).to(self.device).eval()
        embeddings = []

        for i, chunk in enumerate(chunks):
            # Preprocess
            pv = preprocess_frames(chunk.raw_frames, cfg.frames_per_chunk)
            av = preprocess_audio(chunk.audio_segment, cfg.audio_sr)

            pv = pv.to(self.device)
            av = av.to(self.device)
            pos = torch.tensor([i], dtype=torch.long).to(self.device)

            with torch.no_grad():
                z_v = encoder.encode_vision(
                    pv,
                    raw_frames_batch=[chunk.raw_frames],
                    chunk_positions=pos,
                )
                z_a = encoder.encode_audio(av)

            embeddings.append((z_v.cpu(), z_a.cpu(), chunk))

            if (i + 1) % 20 == 0:
                print(f"      encoded {i+1}/{len(chunks)}")

        # Free encoder VRAM before indexing
        del encoder
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # ── Phase 4: Index ────────────────────────────────────────────────
        print(f"[4/4] Indexing  → {cfg.qdrant_path}")
        self.index.index_all(embeddings, video_id)

        elapsed = time.time() - t0
        print(f"\n✓ Done  video_id={video_id}  "
              f"{len(chunks)} chunks  {elapsed:.1f}s elapsed")

    # ──────────────────────────────────────────────────────────────────────
    # Querying
    # ──────────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        video_id: Optional[str] = None,
        top_k: int = 5,
        micro_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Encode a text query and retrieve + rerank matching chunks.

        Returns list of dicts:
            score, video_id, start_sec, end_sec, chunk_id, transcript
        """
        query_vec = self.query_encoder.encode(query_text)

        candidates = self.retriever.retrieve(
            query_vec,
            video_id=video_id,
            micro_only=micro_only,
            top_k=self.cfg_top_k(top_k),
        )

        results = self.reranker.rerank(
            query_vec,
            candidates,
            query_text=query_text,
            top_k=top_k,
        )

        return results

    def cfg_top_k(self, top_k: int) -> int:
        # Fetch more candidates than needed so reranker has room to work
        return max(top_k * 4, cfg.micro_top_k)

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _rough_encode_frames(self, frames) -> torch.Tensor:
        """
        Lightweight CLIP ViT-B/32 frame encoder used only for boundary
        detection. Loaded, used, and immediately freed.

        Embeddings are L2-normalized so that downstream cosine-distance
        boundary detection operates on unit vectors consistently — raw
        open_clip encode_image() output is NOT normalized by default.
        """
        import open_clip
        import torch.nn.functional as F
        from PIL import Image

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model.eval()
        embeds = []
        with torch.no_grad():
            for frame in frames:
                t = preprocess(Image.fromarray(frame)).unsqueeze(0)
                e = model.encode_image(t)
                e = F.normalize(e, p=2, dim=-1)
                embeds.append(e.squeeze(0))
        del model
        gc.collect()
        return torch.stack(embeds)   # (N, 512)


# ── CLI convenience ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python pipeline.py index <video.mp4> <video_id>")
        print("  python pipeline.py query <video_id> <query text>")
        sys.exit(1)

    cmd = sys.argv[1]
    pipe = Pipeline()

    if cmd == "index":
        pipe.index_video(sys.argv[2], sys.argv[3])

    elif cmd == "query":
        vid_id = sys.argv[2]
        q = " ".join(sys.argv[3:])
        results = pipe.query(q, video_id=vid_id, top_k=5)
        print(f"\nQuery: '{q}'")
        print(f"{'─'*60}")
        for r in results:
            ts = f"{r['start_sec']:.1f}s – {r['end_sec']:.1f}s"
            print(f"[{r['score']:.3f}] {ts}  {r['transcript'][:80]}")