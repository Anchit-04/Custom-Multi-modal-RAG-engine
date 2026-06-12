# Multimodal Video RAG Engine

Open-source retrieval engine for semantic querying over MP4 video using
joint audio-visual embeddings. No API dependencies — runs fully local.

## Architecture

```
MP4  →  [Whisper + Semantic Chunker]  →  [VideoMAE + AST + Optical Flow]
     →  [Temporal Aggregator]  →  [Qdrant INT8 index]  →  [CLIP text query]
```

Two-level hierarchical index (micro 3–45s, meso 20–270s) with coarse-to-fine
retrieval and cross-attention reranking.

## Setup

```bash
pip install -r requirements.txt
```

ffmpeg must be on your PATH:
```bash
# Ubuntu
sudo apt install ffmpeg
# macOS
brew install ffmpeg
```

## Usage

### Index a video
```python
from pipeline import Pipeline

pipe = Pipeline()
pipe.index_video("lecture.mp4", video_id="lecture_01")
```

### Query
```python
results = pipe.query("man shooting another man", video_id="lecture_01")
for r in results:
    print(f"{r['start_sec']:.1f}s – {r['end_sec']:.1f}s  score={r['score']}")
```

### CLI
```bash
python pipeline.py index  path/to/video.mp4  my_video_id
python pipeline.py query  my_video_id  "person explaining gradient descent"
```

### Benchmark
```bash
python benchmark.py labels.json
```

## VRAM Budget (sequential loading — never simultaneous)

| Phase        | Model              | Peak VRAM |
|--------------|--------------------|-----------|
| Transcription| Whisper-small      | ~1.5 GB   |
| Boundary det.| CLIP ViT-B/32      | ~1.5 GB   |
| Encoding     | VideoMAE-S + AST   | ~5.0 GB   |
| Query        | CLIP text only     | ~0.5 GB   |

Models are loaded, used, and explicitly deleted between phases.

## Key Design Decisions

- **VideoMAE-Small** over ViT: native temporal understanding, trained on video
- **Optical flow gate**: motion-aware fused embedding, runs on CPU, zero VRAM
- **Temporal aggregator**: 2-layer transformer [CLS] over frame sequence
- **Transcript-guided chunking**: Whisper sentence ends as primary boundaries,
  visual cosine distance as confirmation signal
- **INT8 quantization**: ~4× RAM reduction in Qdrant with negligible accuracy loss
- **Overlap injection**: 2s overlap chunks at every boundary to catch spanning queries
