/**
 * [YOU] Cut detection: port of chunking/boundary_detector.py (git tag python-v1).
 *
 * Combines two signals:
 *   1. Sentence ends from the speech model (utterances ending in . ! ?).
 *   2. Peaks in the smoothed cosine distance between consecutive frame vectors.
 *
 * Steps, as in the Python version:
 *   a. visualDistances(): 1 - cosine(frame[i], frame[i+1]), then moving average.
 *   b. adaptiveThreshold(): max(percentile, mean + stdK * std, floor), computed
 *      per clip so talking heads, screen recordings and B-roll all work.
 *   c. Each sentence end becomes a boundary. It is "both" if the distance peaks
 *      above the threshold within ±matchWindowFrames, else "transcript".
 *   d. Strong local maxima above the threshold become "visual" boundaries,
 *      unless a boundary already exists within visualOnlyMinGapSec.
 *   e. Sort by time and drop any boundary closer than dedupeMinGapSec to the previous.
 *
 * Porting notes:
 *   - The Python docstring says visual-only peaks need 1.5x the threshold, but
 *     the code uses 1x. Choose one on purpose and cover it with a test.
 *   - Frames arrive for a single clip, sorted by time. Vectors are L2-normalised,
 *     so cosine similarity is just the dot product.
 *   - Python's confidence values: "both" = min(1, localMax * 2),
 *     "transcript" = 0.5, "visual" = min(1, distance * 1.5).
 */
import type { Boundary, FrameEmbedding, Utterance } from '../types';
import { config, type BoundaryConfig } from '../config';
import { notImplemented } from '../notImplemented';

export function detectBoundaries(
  frames: FrameEmbedding[],
  utterances: Utterance[],
  opts: BoundaryConfig = config.boundary,
): Boundary[] {
  notImplemented('detectBoundaries');
}

/** Smoothed distance between frame i and frame i+1. Length is frames.length - 1. */
export function visualDistances(frames: FrameEmbedding[], smoothWindow: number): number[] {
  notImplemented('visualDistances');
}

/** Per-clip cut threshold derived from the distance signal itself. */
export function adaptiveThreshold(
  distances: number[],
  opts: Pick<BoundaryConfig, 'percentile' | 'stdK' | 'floor'> = config.boundary,
): number {
  notImplemented('adaptiveThreshold');
}
