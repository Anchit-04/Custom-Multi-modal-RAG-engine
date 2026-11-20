/**
 * [YOU] Turns boundaries into segments: port of chunking/duration_enforcer.py
 * and the meso grouping from indexing/hierarchal_index.py (git tag python-v1).
 *
 * enforceDuration():
 *   - Cut points are 0, every boundary time, and the clip duration.
 *   - Merge cuts closer than minSec to the previous kept cut.
 *   - Split any stretch longer than maxSec into equal parts.
 *   - A trailing piece shorter than minSec is merged into the one before it.
 *   - Each segment gets keyframeT (its midpoint is fine) and the transcript of
 *     every utterance that overlaps it.
 *
 * groupMeso():
 *   - Every mesoGroupSize consecutive micro segments become one meso segment
 *     spanning them, with their transcripts joined.
 *
 * Unlike python-v1, segments no longer carry raw frames or audio: media stays in
 * the file and is read again only when needed.
 */
import type { Boundary, ClipId, Seconds, Segment, Utterance } from '../types';
import { config, type SegmentConfig } from '../config';
import { notImplemented } from '../notImplemented';

export function enforceDuration(
  clipId: ClipId,
  durationSec: Seconds,
  boundaries: Boundary[],
  utterances: Utterance[],
  opts: SegmentConfig = config.segment,
): Segment[] {
  notImplemented('enforceDuration');
}

export function groupMeso(micro: Segment[], groupSize: number = config.segment.mesoGroupSize): Segment[] {
  notImplemented('groupMeso');
}
