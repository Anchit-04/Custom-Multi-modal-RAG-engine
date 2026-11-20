import type { Seconds, Utterance } from '../../core/types';
import type { LibraryItem } from '../useLibrary';
import { formatTime } from '../format';

interface Props {
  item: LibraryItem | null;
  currentTime: Seconds;
  onSeek(t: Seconds): void;
}

export function TranscriptView({ item, currentTime, onSeek }: Props) {
  if (!item) return null;

  if (item.status !== 'done') {
    return <p className="muted small">The transcript appears here once this clip is processed.</p>;
  }
  if (item.utterances.length === 0) {
    return <p className="muted small">No speech found in this clip.</p>;
  }

  return (
    <ol className="transcript">
      {item.utterances.map((u, i) => (
        <Line key={i} utterance={u} active={currentTime >= u.start && currentTime < u.end} onSeek={onSeek} />
      ))}
    </ol>
  );
}

function Line({ utterance, active, onSeek }: { utterance: Utterance; active: boolean; onSeek(t: Seconds): void }) {
  return (
    <li className={active ? 'is-active' : undefined}>
      <button type="button" className="timestamp" onClick={() => onSeek(utterance.start)}>
        {formatTime(utterance.start)}
      </button>
      <span>
        {utterance.words.map((w, i) => (
          <span key={i} className="word" title={`${w.start.toFixed(2)}s`} onClick={() => onSeek(w.start)}>
            {w.text}{' '}
          </span>
        ))}
      </span>
    </li>
  );
}
