import { useState } from 'react';
import type { ClipId, Seconds } from '../core/types';
import { isProfileName, pickProfile, type ProfileName } from '../models/device';
import { MODEL_PLANS } from '../models/profiles';
import { useIngestWorker } from './useIngestWorker';
import { useLibrary } from './useLibrary';
import { DeviceCard } from './components/DeviceCard';
import { ModelPlanCard } from './components/ModelPlanCard';
import { LibraryPanel } from './components/LibraryPanel';
import { Player, type SeekRequest } from './components/Player';
import { SearchPanel } from './components/SearchPanel';
import { TranscriptView } from './components/TranscriptView';

/** ?profile=lite|standard|max forces a profile, e.g. to test lite on a strong machine. */
function profileOverride(): ProfileName | null {
  const value = new URLSearchParams(window.location.search).get('profile');
  return isProfileName(value) ? value : null;
}

export function App() {
  const worker = useIngestWorker();
  const override = profileOverride();
  const profile = worker.status === 'ready' ? (override ?? pickProfile(worker.caps)) : null;
  const library = useLibrary(worker, profile);

  const [selectedId, setSelectedId] = useState<ClipId | null>(null);
  const [seek, setSeek] = useState<SeekRequest | null>(null);
  const [currentTime, setCurrentTime] = useState<Seconds>(0);

  const selected = library.items.find((i) => i.clip.id === selectedId) ?? library.items[0] ?? null;

  const select = (id: ClipId) => {
    setSelectedId(id);
    setCurrentTime(0);
  };
  const jump = (clipId: ClipId, t: Seconds) => {
    setSelectedId(clipId);
    setSeek((prev) => ({ t, n: (prev?.n ?? 0) + 1 }));
  };

  return (
    <main className="page">
      <header className="masthead">
        <h1>framegrep</h1>
        <p className="tagline">Search what's said in your footage and jump to the exact word. Runs in your browser.</p>
      </header>

      {worker.status === 'starting' && <p className="muted">Starting the background worker…</p>}
      {worker.status === 'error' && (
        <p className="card error" role="alert">
          Worker failed: {worker.message}
        </p>
      )}

      <SearchPanel items={library.items} onPick={jump} />

      <div className="workspace">
        <LibraryPanel
          library={library}
          selectedId={selected?.clip.id ?? null}
          onSelect={select}
          disabled={worker.status !== 'ready'}
        />
        <section className="card viewer" aria-label="Player">
          <h2>{selected ? selected.clip.name : 'Player'}</h2>
          <Player file={selected?.file ?? null} seek={seek} onTime={setCurrentTime} />
          <TranscriptView item={selected} currentTime={currentTime} onSeek={(t) => selected && jump(selected.clip.id, t)} />
        </section>
      </div>

      {worker.status === 'ready' && profile && (
        <details className="system">
          <summary>
            Device and models: <strong>{profile}</strong> profile
          </summary>
          <div className="grid">
            <DeviceCard caps={worker.caps} profile={profile} overridden={override !== null} />
            <ModelPlanCard profile={profile} plan={MODEL_PLANS[profile]} />
          </div>
        </details>
      )}
    </main>
  );
}
