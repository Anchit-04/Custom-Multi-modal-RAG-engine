/**
 * Search box and results. The ranking itself is TextIndex (core/index/textIndex.ts);
 * this file only builds the index from transcripts and shows what it returns.
 */
import { useMemo, useState } from 'react';
import type { ClipId, Hit, Seconds, Utterance } from '../../core/types';
import { TextIndex, type TextDoc } from '../../core/index/textIndex';
import type { LibraryItem } from '../useLibrary';
import { errorMessage, formatTime } from '../format';

interface Props {
  items: LibraryItem[];
  onPick(clipId: ClipId, t: Seconds): void;
}

type IndexState =
  | { kind: 'ready'; index: TextIndex; byDoc: Map<string, { item: LibraryItem; utterance: Utterance }> }
  | { kind: 'unavailable'; reason: string };

function buildIndex(items: LibraryItem[]): IndexState {
  const byDoc = new Map<string, { item: LibraryItem; utterance: Utterance }>();
  try {
    const index = new TextIndex('speech');
    for (const item of items) {
      item.utterances.forEach((utterance, i) => {
        const doc: TextDoc = {
          id: `${item.clip.id}#${i}`,
          clipId: item.clip.id,
          start: utterance.start,
          end: utterance.end,
          text: utterance.text,
          words: utterance.words,
        };
        index.add(doc);
        byDoc.set(doc.id, { item, utterance });
      });
    }
    return { kind: 'ready', index, byDoc };
  } catch (err) {
    return { kind: 'unavailable', reason: errorMessage(err) };
  }
}

export function SearchPanel({ items, onPick }: Props) {
  const [query, setQuery] = useState('');
  const [phrase, setPhrase] = useState(false);

  const searchable = useMemo(() => items.filter((i) => i.status === 'done'), [items]);
  const state = useMemo(() => buildIndex(searchable), [searchable]);

  const result = useMemo((): { hits: Hit[]; error?: string } => {
    if (state.kind !== 'ready' || !query.trim()) return { hits: [] };
    try {
      return { hits: state.index.search(query, { phrase, limit: 30 }) };
    } catch (err) {
      return { hits: [], error: errorMessage(err) };
    }
  }, [state, query, phrase]);

  const notice =
    state.kind === 'unavailable'
      ? `Search turns on once TextIndex is implemented (src/core/index/textIndex.ts): ${state.reason}`
      : result.error;

  return (
    <section className="search" aria-label="Search">
      <div className="search-row">
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={searchable.length ? 'Search what was said…' : 'Add footage to search it'}
          aria-label="Search spoken words"
        />
        <label className="toggle">
          <input type="checkbox" checked={phrase} onChange={(e) => setPhrase(e.target.checked)} /> Exact phrase
        </label>
      </div>

      {notice && query.trim() && <p className="notice small">{notice}</p>}

      {state.kind === 'ready' && query.trim() && !result.error && (
        <ol className="results">
          {result.hits.length === 0 && <li className="muted small">No matches.</li>}
          {result.hits.map((hit) => {
            const source = state.byDoc.get(hit.docId);
            return (
              <li key={`${hit.docId}@${hit.start}`}>
                <button type="button" className="result" onClick={() => onPick(hit.clipId, hit.start)}>
                  <span className="result-time">{formatTime(hit.start)}</span>
                  <span className="result-clip">{source?.item.clip.name ?? hit.clipId}</span>
                  <span className="result-text">{source?.utterance.text ?? hit.matched}</span>
                  {hit.matched && <span className="chip">said “{hit.matched}”</span>}
                </button>
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}
