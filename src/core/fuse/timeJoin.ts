/**
 * [YOU] Combining hits by time: "said X while Y was on screen". Milestone M4.
 *
 * joinWithin():
 *   - Pair hits from `a` and `b` on the same clip whose spans overlap or lie
 *     within windowSec of each other.
 *   - Each pair becomes one FusedHit spanning both, carrying both hits.
 *   - Merge overlapping results so one moment isn't returned twice.
 *   - Sort per clip by start and sweep, rather than comparing every pair.
 *
 * excludeWithin():
 *   - Drop results that have any `b` hit within windowSec (for NOT).
 */
import type { FusedHit, Hit, Seconds } from '../types';
import { notImplemented } from '../notImplemented';

export function joinWithin(a: Array<Hit | FusedHit>, b: Array<Hit | FusedHit>, windowSec: Seconds): FusedHit[] {
  notImplemented('joinWithin');
}

export function excludeWithin(results: FusedHit[], b: Hit[], windowSec: Seconds): FusedHit[] {
  notImplemented('excludeWithin');
}
