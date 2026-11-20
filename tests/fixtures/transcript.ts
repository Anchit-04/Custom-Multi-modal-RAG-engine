/**
 * A small hand-written transcript: a talking head with a retake, a pricing
 * line, and some punctuation. Word timings are evenly spaced and fake, but
 * consistent, so tests can check exact timestamps.
 */
import type { TextDoc } from '../../src/core/index/textIndex';
import type { Word } from '../../src/core/types';

function words(text: string, start: number, secPerWord = 0.4): Word[] {
  return text.split(/\s+/).map((w, i) => ({
    text: w,
    start: +(start + i * secPerWord).toFixed(2),
    end: +(start + (i + 1) * secPerWord).toFixed(2),
  }));
}

function utterance(id: string, text: string, start: number): TextDoc {
  const w = words(text, start);
  return { id, clipId: 'take-01', start, end: w[w.length - 1]!.end, text, words: w };
}

export const talkingHead: TextDoc[] = [
  utterance('u1', 'Hi everyone, today we are launching framegrep.', 0),
  utterance('u2', 'Today we are launching, wait, let me start over.', 4),
  utterance('u3', 'Today we are launching framegrep, a search engine for footage.', 8),
  utterance('u4', 'Pricing starts at ten dollars a month.', 14),
  utterance('u5', 'The pricing page has every plan on it.', 18),
];

/** OCR text from keyframes: no word timings, so hits span the whole frame window. */
export const screenText: TextDoc[] = [
  { id: 's1', clipId: 'screen-01', start: 2, end: 2.5, text: 'Dashboard  Projects  Settings' },
  { id: 's2', clipId: 'screen-01', start: 9, end: 9.5, text: 'Pricing: Starter $10/mo, Pro $29/mo' },
];
