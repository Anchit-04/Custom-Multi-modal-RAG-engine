import { useEffect, useState } from 'react';
import { createIngestWorker } from '../worker/client';
import { isProfileName, pickProfile, type DeviceCaps, type ProfileName } from '../models/device';
import { MODEL_PLANS } from '../models/profiles';
import { DeviceCard } from './components/DeviceCard';
import { ModelPlanCard } from './components/ModelPlanCard';

type WorkerState = { status: 'starting' } | { status: 'ready'; caps: DeviceCaps } | { status: 'error'; message: string };

/** ?profile=lite|standard|max forces a profile, e.g. to test lite on a strong machine. */
function profileOverride(): ProfileName | null {
  const value = new URLSearchParams(window.location.search).get('profile');
  return isProfileName(value) ? value : null;
}

export function App() {
  const [worker, setWorker] = useState<WorkerState>({ status: 'starting' });

  useEffect(() => {
    const { api, terminate } = createIngestWorker();
    let cancelled = false;
    (async () => {
      try {
        if ((await api.ping()) !== 'pong') throw new Error('Worker did not answer');
        const caps = await api.probeDevice();
        if (!cancelled) setWorker({ status: 'ready', caps });
      } catch (err) {
        if (!cancelled) setWorker({ status: 'error', message: err instanceof Error ? err.message : String(err) });
      }
    })();
    return () => {
      cancelled = true;
      terminate();
    };
  }, []);

  const override = profileOverride();
  const profile = worker.status === 'ready' ? (override ?? pickProfile(worker.caps)) : null;

  return (
    <main className="page">
      <header className="masthead">
        <h1>framegrep</h1>
        <p className="tagline">Search what's said, what's on screen and what's shown in your footage. Runs in your browser.</p>
      </header>

      <section className="search-placeholder" aria-label="Search">
        <input type="search" placeholder={'say:"launch" screen:pricing   (search arrives in M1)'} disabled />
        <button type="button" disabled>
          Choose footage folder
        </button>
      </section>

      <div className="grid">
        {worker.status === 'starting' && <p className="card muted">Starting the background worker…</p>}
        {worker.status === 'error' && (
          <p className="card error" role="alert">
            Worker failed: {worker.message}
          </p>
        )}
        {worker.status === 'ready' && profile && (
          <>
            <DeviceCard caps={worker.caps} profile={profile} overridden={override !== null} />
            <ModelPlanCard profile={profile} plan={MODEL_PLANS[profile]} />
          </>
        )}
      </div>
    </main>
  );
}
