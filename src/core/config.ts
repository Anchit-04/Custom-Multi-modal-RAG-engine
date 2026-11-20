/**
 * Tunables. Values marked "python-v1" are carried over from config.py and
 * boundary_detector.py in the original Python version (git tag python-v1).
 */
export const config = {
  ingest: {
    /** Frames sampled per second for visual vectors and cut detection (python-v1: 2). */
    frameFps: 2,
    /** Speech models expect 16 kHz mono (python-v1). */
    audioSampleRate: 16_000,
  },

  boundary: {
    /** Moving-average window over frame-to-frame cosine distance (python-v1). */
    smoothWindow: 4,
    /** Threshold = max(percentile, mean + stdK * std, floor) (python-v1). */
    percentile: 90,
    stdK: 2.5,
    floor: 0.15,
    /** A sentence end is "both" if a visual peak lies within this many frames (python-v1). */
    matchWindowFrames: 5,
    /** Visual-only cuts are skipped if a boundary already exists this close (python-v1). */
    visualOnlyMinGapSec: 2.0,
    /** After sorting, drop boundaries closer than this to the previous one (python-v1). */
    dedupeMinGapSec: 1.0,
  },

  segment: {
    /** python-v1 */
    minSec: 3.0,
    maxSec: 45.0,
    /** Micro segments per meso segment (python-v1). */
    mesoGroupSize: 6,
  },

  search: {
    /** Candidates taken from the 1-bit vector scan before precise re-ranking. */
    binaryCandidates: 200,
    /** How many results get re-ranked with the precise model. */
    rerankTopK: 20,
    /** Standard reciprocal rank fusion constant. */
    rrfK: 60,
    /** Default window for "A and B at the same moment". */
    joinWindowSec: 3,
  },
} as const;

export type Config = typeof config;
export type BoundaryConfig = Config['boundary'];
export type SegmentConfig = Config['segment'];
