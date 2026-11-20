# Architecture

framegrep answers one question fast and precisely: *where in this footage does
X happen?* It splits that into signals that can each be indexed cheaply, then
recombines them by time.

## Overview

```
footage folder (read in place, File System Access API)
        │
        ▼
┌──────────────────── web worker ────────────────────┐
│ Tier 1, at import           Tier 2, on demand        │
│  shots + keyframes           SigLIP 2 keyframe vectors│
│  speech → words + times      Florence-2 captions      │
│  MobileCLIP2 frame vectors   EmbeddingGemma vectors   │
│  PP-OCRv5 screen text                                │
└──────────────┬───────────────────────────┬──────────┘
               ▼                           ▼
 indexes:  speech words · screen text · 1-bit vectors · precise vectors · text vectors
               │
               ▼
 query box → parser → planner → search each index → time join + rank fusion → re-rank → results
```

## The pieces

**Tier 1 (import)** makes footage searchable quickly: sample frames, compute
cheap visual vectors, detect cuts, transcribe with word timings, and OCR one
keyframe per shot.

**Tier 2 (on demand)** runs expensive models only where searches land: precise
vectors for re-ranking, captions, and text vectors for meaning-based search.

**Indexes**
- *Speech words* and *screen text*: inverted text indexes with BM25 ranking and
  phrase search. Speech hits narrow to the exact words.
- *1-bit vectors*: the sign of each dimension, packed 32 per word. 512 dims =
  64 bytes per frame. Hamming distance scan, then top candidates go to:
- *Precise vectors*: float keyframe vectors from a stronger model, for re-ranking.
- *Text vectors*: meaning-based search over speech and captions.

**Query path**
1. `parser.ts` turns `say:"launch" screen:pricing` into a query tree.
2. `planner.ts` maps each field to its indexes.
3. Each index is searched.
4. `timeJoin.ts` combines AND / NEAR / NOT by time; `rrf.ts` merges rankings
   from different indexes by rank, since their scores aren't comparable.
5. `rerank.ts` re-scores the top results with precise vectors.

## Code boundaries

| Layer | Rule |
|---|---|
| `src/core` | Pure functions over the types in `core/types.ts`. No DOM, no models, no async. Tested in Node. |
| `src/models` | One interface per job; implementations are swappable per profile. |
| `src/worker` | Owns every model. The UI talks to it through Comlink. |
| `src/media` | Browser media input and output on the main thread. |
| `src/app` | React UI. No model or index logic. |

## Device profiles

`models/device.ts` checks WebGPU support, fp16 shaders, GPU vendor and
architecture, buffer limits and RAM, then picks `lite`, `standard` or `max`.
`models/profiles.ts` lists the models per profile. Each model id is confirmed
in the milestone that first uses it.

## What it does not cover yet

- Non-speech audio (music, sound effects). CLAP embeddings are the natural
  extension.
- Importing hundreds of gigabytes is limited by the local GPU. Searching that
  much is not. A local helper or server GPU could take over imports at scale
  while keeping the same index format.
