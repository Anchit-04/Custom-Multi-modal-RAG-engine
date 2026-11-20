/**
 * [YOU] Query language parser: the "grep" in framegrep. Milestone M4.
 *
 * Proposed syntax (yours to change, as long as the tests agree):
 *
 *   launch                      any field, single word
 *   "we are launching"          any field, exact phrase
 *   say:launch                  spoken words only
 *   screen:"pricing page"       on-screen text only
 *   shows:"person with phone"   visual (vectors, plus captions)
 *   say:launch screen:pricing   both, at the same moment (implicit AND)
 *   a OR b                      either
 *   -screen:draft               exclude (also: NOT screen:draft)
 *   (a OR b) c                  grouping
 *   say:launch NEAR/10 shows:logo   within 10 seconds of each other
 *
 * Output is a QueryNode tree (see types.ts). Unknown field names should become
 * plain text rather than errors, so typing never "breaks" the search box.
 * Suggested approach: a small tokenizer, then recursive descent with
 * precedence NOT > NEAR > AND > OR.
 */
import type { QueryNode } from '../types';
import { notImplemented } from '../notImplemented';

export function parseQuery(input: string): QueryNode {
  notImplemented('parseQuery');
}
