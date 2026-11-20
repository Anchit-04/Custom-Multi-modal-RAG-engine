/**
 * [YOU] Text index used for speech, on-screen text and captions. Milestone M1.
 *
 * Requirements (pinned down by textIndex.test.ts, currently skipped):
 *   - Case-insensitive; punctuation ignored ("Pricing," matches "pricing").
 *   - Ranked by BM25 (k1 = 1.2, b = 0.75 are the usual defaults).
 *   - Phrase search: words must appear consecutively and in order.
 *   - When a document has word timings, a hit's start/end narrow to the matched
 *     words, so the player can jump to the exact word. Otherwise the hit spans
 *     the whole document.
 *   - Every hit carries the index's signal, the document id, and the matched text.
 *
 * Design hints:
 *   - Inverted index: term -> list of (doc number, positions). Positions make
 *     phrase search cheap.
 *   - Keep token positions aligned with doc.words so a phrase match maps
 *     straight to word timings.
 *   - search() is called on every keystroke, so do the work in add().
 */
import type { ClipId, Hit, Seconds, Signal, Word } from '../types';
import { notImplemented } from '../notImplemented';

export interface TextDoc {
  id: string;
  clipId: ClipId;
  start: Seconds;
  end: Seconds;
  text: string;
  /** Word timings, when available (speech). One entry per token of `text`. */
  words?: Word[];
}

export interface TextSearchOptions {
  /** Maximum number of hits. Default 50. */
  limit?: number;
  /** Treat the query as an exact phrase. Default false (match any term, ranked). */
  phrase?: boolean;
}

export class TextIndex {
  constructor(readonly signal: Exclude<Signal, 'visual'>) {}

  add(doc: TextDoc): void {
    notImplemented('TextIndex.add');
  }

  get size(): number {
    return notImplemented('TextIndex.size');
  }

  search(query: string, opts: TextSearchOptions = {}): Hit[] {
    notImplemented('TextIndex.search');
  }
}
