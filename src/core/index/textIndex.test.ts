/**
 * Spec for TextIndex (milestone M1). Skipped until you implement it:
 * change `describe.skip` to `describe` and run `npm run test:watch`.
 */
import { describe, expect, it } from 'vitest';
import { TextIndex } from './textIndex';
import { screenText, talkingHead } from '../../../tests/fixtures/transcript';

function speechIndex() {
  const index = new TextIndex('speech');
  talkingHead.forEach((doc) => index.add(doc));
  return index;
}

describe.skip('TextIndex', () => {
  it('starts empty and returns nothing', () => {
    const index = new TextIndex('speech');
    expect(index.size).toBe(0);
    expect(index.search('anything')).toEqual([]);
  });

  it('counts added documents', () => {
    expect(speechIndex().size).toBe(talkingHead.length);
  });

  it('is case-insensitive and ignores punctuation', () => {
    const ids = speechIndex()
      .search('FRAMEGREP')
      .map((h) => h.docId);
    expect(ids.sort()).toEqual(['u1', 'u3']);
  });

  it('returns nothing for a word that never appears', () => {
    expect(speechIndex().search('kubernetes')).toEqual([]);
  });

  it('ranks with BM25: a rarer term outweighs a common one', () => {
    // "launching" is in 3 docs, "footage" only in u3, so u3 ranks first.
    const hits = speechIndex().search('launching footage');
    expect(hits[0]?.docId).toBe('u3');
    expect(hits.map((h) => h.score)).toEqual([...hits.map((h) => h.score)].sort((a, b) => b - a));
  });

  it('phrase search needs consecutive words in order', () => {
    const index = speechIndex();
    expect(index.search('pricing starts', { phrase: true }).map((h) => h.docId)).toEqual(['u4']);
    expect(index.search('starts pricing', { phrase: true })).toEqual([]);
  });

  it('narrows a phrase hit to the matched words when timings exist', () => {
    // u4 starts at 14s with 0.4s per word: "Pricing"=14.0-14.4, "starts"=14.4-14.8
    const [hit] = speechIndex().search('pricing starts', { phrase: true });
    expect(hit?.start).toBeCloseTo(14.0);
    expect(hit?.end).toBeCloseTo(14.8);
  });

  it('uses the whole document span when there are no word timings', () => {
    const index = new TextIndex('screen');
    screenText.forEach((doc) => index.add(doc));
    const [hit] = index.search('pricing');
    expect(hit).toMatchObject({ docId: 's2', clipId: 'screen-01', start: 9, end: 9.5 });
  });

  it('labels hits with the index signal and the matched text', () => {
    const [hit] = speechIndex().search('pricing starts', { phrase: true });
    expect(hit?.signal).toBe('speech');
    expect(hit?.matched?.toLowerCase()).toContain('pricing starts');
  });

  it('respects the limit', () => {
    expect(speechIndex().search('today', { limit: 2 })).toHaveLength(2);
  });
});
