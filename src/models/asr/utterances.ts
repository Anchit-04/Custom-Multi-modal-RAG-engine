/**
 * Groups timed words into utterances (sentence-like runs of speech).
 *
 * An utterance ends at sentence punctuation, at a pause, or when it gets long.
 * Its text is the words joined by single spaces, so splitting `text` on
 * whitespace gives back exactly one token per entry in `words`; TextIndex
 * relies on that to map a phrase match to word timings.
 */
import type { ClipId, Seconds, Utterance, Word } from '../../core/types';

export interface GroupingOptions {
  /** A silence longer than this starts a new utterance. */
  maxPauseSec: Seconds;
  /** Hard cap so a run-on monologue still produces searchable pieces. */
  maxWords: number;
}

const DEFAULTS: GroupingOptions = { maxPauseSec: 0.8, maxWords: 40 };
const SENTENCE_END = /[.!?]["')\]]*$/;

export function groupIntoUtterances(clipId: ClipId, words: Word[], opts: GroupingOptions = DEFAULTS): Utterance[] {
  const utterances: Utterance[] = [];
  let current: Word[] = [];

  const flush = () => {
    const first = current[0];
    const last = current.at(-1);
    if (first && last) {
      utterances.push({
        clipId,
        text: current.map((w) => w.text).join(' '),
        start: first.start,
        end: last.end,
        words: current,
      });
    }
    current = [];
  };

  const tokens = words.flatMap(splitMultiWord);
  tokens.forEach((word, i) => {
    current.push(word);
    const next = tokens[i + 1];
    const pause = next ? next.start - word.end : Infinity;
    if (SENTENCE_END.test(word.text) || pause > opts.maxPauseSec || current.length >= opts.maxWords) flush();
  });
  flush();
  return utterances;
}

/** Guarantees one whitespace-free token per Word, sharing the time span if a chunk held several. */
function splitMultiWord(word: Word): Word[] {
  const parts = word.text.split(/\s+/).filter(Boolean);
  if (parts.length <= 1) return parts.length ? [{ ...word, text: parts[0]! }] : [];
  const step = (word.end - word.start) / parts.length;
  return parts.map((text, i) => ({ ...word, text, start: word.start + i * step, end: word.start + (i + 1) * step }));
}
