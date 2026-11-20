# framegrep

**grep for your footage.** Point it at a folder of raw video and search what's
said, what's on screen and what's shown, combine them like grep, and jump to the
exact second. Everything runs in the browser on the user's own GPU: no upload,
no server.

```
say:"launch" screen:pricing          said "launch" while the pricing page was on screen
shows:"person holding a phone" -screen:draft
say:"thank you" NEAR/10 shows:logo
```

> **Status: M0, scaffold.** The app starts, checks the device and picks a model
> profile. Search lands in M1. See [the roadmap](#roadmap).

## Why

Text is searchable by default; video isn't. Embedding-only video search is
either too slow or too inaccurate for an editing workflow. framegrep treats each
signal on its own terms:

- **Exact indexes** for spoken words (with per-word timestamps) and on-screen
  text (OCR): the parts that can be grepped are grepped.
- **1-bit visual vectors** for a millisecond first pass over hundreds of hours,
  then **precise re-ranking** of the top candidates only.
- **Two tiers**: footage is searchable minutes after import; expensive models
  run later, only where searches land.
- **Time-aligned fusion**, so one query can say "this was said while that was
  on screen".

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Runs on weak hardware, scales up on strong hardware

The app checks the device at startup and picks a profile. Force one with
`?profile=lite|standard|max`.

| Profile | Typical device | Speech | Visual | Re-rank | Captions |
|---|---|---|---|---|---|
| lite | Older integrated GPU, or CPU only | Whisper base | MobileCLIP2-S0 | off | off |
| standard | Newer integrated or mid-range GPU | Parakeet TDT v3 | MobileCLIP2-S2 | SigLIP 2 base | Florence-2 |
| max | Strong discrete GPU | Parakeet TDT v3 | MobileCLIP2-B | SigLIP 2 large | Florence-2 |

Development baseline: a 2019 laptop with Intel UHD graphics (gen-9). If it works
there, it works anywhere.

## Getting started

Requires Node 22+ and a Chromium browser (Chrome or Edge) with WebGPU.

```bash
npm install
npm run dev        # http://localhost:5173
npm test           # unit tests
npm run typecheck
npm run build
```

## Project layout

```
src/
  core/      pure TypeScript: segmenting, indexes, query parser, fusion (no DOM, no models)
  media/     folder access, audio decoding, frame sampling           (M1–M2)
  models/    device check, profiles, one wrapper per model
  worker/    background thread for all model work
  store/     on-disk indexes (OPFS) and the clip catalog             (M6)
  app/       React UI
bench/       accuracy and scaling benchmarks                         (M7)
tests/       shared fixtures
docs/        architecture and benchmark write-ups
```

## Roadmap

| | Milestone | Demo at the end |
|---|---|---|
| ✅ M0 | Scaffold | Device profile detected, worker running, tests green |
| M1 | Grep for speech | Type a phrase, jump to the exact word |
| M2 | Visual search | "Person holding a phone" returns shots |
| M3 | Screen text | Find the frame where "pricing" is on screen |
| M4 | Query language + fusion | `say:"launch" screen:dashboard` |
| M5 | Precision tier | Accuracy rises with re-ranking and captions |
| M6 | Persistence + scale | Reopen the tab and search instantly |
| M7 | Benchmarks | Recall@k, latency vs library size, bytes per hour |

## History

This project started as a Python "video RAG" engine (VideoMAE + AST + CLIP,
Qdrant). That version is preserved at the git tag
[`python-v1`](../../tree/python-v1).
