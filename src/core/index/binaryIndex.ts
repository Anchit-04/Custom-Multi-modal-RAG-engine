/**
 * [YOU] 1-bit vector index: the fast first pass of visual search. Milestone M2.
 *
 * quantize():
 *   - Keep only the sign of each dimension: bit = 1 if value > 0.
 *   - Pack 32 dimensions per Uint32 word, so 512 dims -> 16 words (64 bytes),
 *     32x smaller than float32.
 *
 * hamming():
 *   - Count the bits that differ: popcount(a XOR b), summed over words.
 *   - JS has no popcount builtin; the classic bit-twiddling version is fast.
 *
 * BinaryIndex:
 *   - Store all codes in one growing Uint32Array, not one array per vector:
 *     a linear scan over contiguous memory is what makes this fast.
 *   - search() returns the k smallest distances, nearest first.
 *
 * Why it works: for normalised vectors, Hamming distance between sign codes
 * tracks the angle between them, so close codes are close vectors. The few
 * hundred best candidates are then re-scored with full-precision vectors.
 */
import { notImplemented } from '../notImplemented';

export function quantize(vector: Float32Array): Uint32Array {
  notImplemented('quantize');
}

export function hamming(a: Uint32Array, b: Uint32Array): number {
  notImplemented('hamming');
}

export interface BinaryMatch {
  id: string;
  distance: number;
}

export class BinaryIndex {
  constructor(readonly dim: number) {}

  add(id: string, vector: Float32Array): void {
    notImplemented('BinaryIndex.add');
  }

  get size(): number {
    return notImplemented('BinaryIndex.size');
  }

  search(query: Float32Array, k: number): BinaryMatch[] {
    notImplemented('BinaryIndex.search');
  }
}
