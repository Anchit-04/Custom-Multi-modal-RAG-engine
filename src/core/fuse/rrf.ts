/**
 * [YOU] Reciprocal rank fusion: merging rankings from different indexes. Milestone M4.
 *
 * Scores from BM25, Hamming distance and cosine similarity aren't comparable,
 * so fuse by rank instead: score = sum over lists of 1 / (k + rank), rank from 1.
 *
 *   - Hits from different lists that cover the same moment (same clip, spans
 *     overlapping or within mergeWindowSec) count as the same result, and their
 *     hits are collected on the FusedHit.
 *   - Output is sorted by fused score, highest first.
 */
import type { FusedHit, Hit, Seconds } from '../types';
import { config } from '../config';
import { notImplemented } from '../notImplemented';

export function reciprocalRankFusion(
  lists: Hit[][],
  k: number = config.search.rrfK,
  mergeWindowSec: Seconds = 1,
): FusedHit[] {
  notImplemented('reciprocalRankFusion');
}
