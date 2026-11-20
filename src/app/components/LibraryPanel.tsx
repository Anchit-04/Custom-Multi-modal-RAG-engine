import { useState, type ChangeEvent } from 'react';
import type { ClipId } from '../../core/types';
import { canPickFolder, fromFileList, pickFolder, type PickedFile } from '../../media/folder';
import type { Library, LibraryItem } from '../useLibrary';
import { errorMessage, formatBytes, formatTime } from '../format';

interface Props {
  library: Library;
  selectedId: ClipId | null;
  onSelect(id: ClipId): void;
  disabled: boolean;
}

const STATUS_LABEL: Record<LibraryItem['status'], string> = {
  queued: 'Queued',
  decoding: 'Reading audio…',
  transcribing: 'Transcribing…',
  done: 'Searchable',
  error: 'Failed',
};

export function LibraryPanel({ library, selectedId, onSelect, disabled }: Props) {
  const [pickError, setPickError] = useState<string | null>(null);

  const add = async (picked: PickedFile[] | null) => {
    setPickError(null);
    if (!picked) return;
    if (picked.length === 0) {
      setPickError('No video files found (looking for .mp4, .mov, .webm, .mkv, .m4v).');
      return;
    }
    await library.addFiles(picked);
  };

  const onFolder = () => pickFolder().then(add, (err) => setPickError(errorMessage(err)));
  const onFiles = (e: ChangeEvent<HTMLInputElement>) => {
    void add(fromFileList(e.target.files));
    e.target.value = ''; // allow picking the same files again
  };

  return (
    <section className="card library" aria-label="Footage">
      <div className="library-head">
        <h2>Footage</h2>
        <div className="library-actions">
          {canPickFolder() && (
            <button type="button" onClick={onFolder} disabled={disabled}>
              Choose folder
            </button>
          )}
          <label className={`button ${disabled ? 'is-disabled' : ''}`}>
            Add files
            <input type="file" accept="video/*" multiple onChange={onFiles} disabled={disabled} hidden />
          </label>
        </div>
      </div>

      {pickError && <p className="error small">{pickError}</p>}
      {library.download && <DownloadBar download={library.download} />}

      {library.items.length === 0 ? (
        <p className="muted small">Add a few clips to make them searchable. Nothing is uploaded.</p>
      ) : (
        <ul className="clip-list">
          {library.items.map((item) => (
            <li key={item.clip.id}>
              <button
                type="button"
                className={`clip ${item.clip.id === selectedId ? 'is-selected' : ''}`}
                onClick={() => onSelect(item.clip.id)}
              >
                <span className="clip-name">{item.clip.name}</span>
                <span className="clip-meta">
                  {formatTime(item.clip.durationSec)} · {formatBytes(item.clip.sizeBytes)}
                </span>
                <span className={`clip-status status-${item.status}`}>
                  {STATUS_LABEL[item.status]}
                  {item.status === 'done' && item.speech && <SpeechStats item={item} />}
                </span>
                {item.error && <span className="error small">{item.error}</span>}
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function SpeechStats({ item }: { item: LibraryItem }) {
  const speech = item.speech!;
  const speed = speech.elapsedSec > 0 ? speech.audioSec / speech.elapsedSec : 0;
  const model = speech.model.split('/').at(-1);
  return (
    <span className="muted">
      {' '}
      · {item.utterances.length} lines · {speed.toFixed(1)}× real time · {model} on {speech.device}
    </span>
  );
}

function DownloadBar({ download }: { download: NonNullable<Library['download']> }) {
  const pct = download.total > 0 ? Math.round((download.loaded / download.total) * 100) : 0;
  return (
    <div className="download">
      <span className="small">
        Downloading {download.model.split('/').at(-1)} (first run only): {formatBytes(download.loaded)} of{' '}
        {formatBytes(download.total)}
      </span>
      <progress max={100} value={pct} />
    </div>
  );
}
