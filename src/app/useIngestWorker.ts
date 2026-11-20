import { useEffect, useState } from 'react';
import type { Remote } from 'comlink';
import type { DeviceCaps } from '../models/device';
import type { IngestApi } from '../worker/ingest.worker';
import { createIngestWorker } from '../worker/client';
import { errorMessage } from './format';

export type WorkerState =
  | { status: 'starting' }
  | { status: 'ready'; caps: DeviceCaps; api: Remote<IngestApi> }
  | { status: 'error'; message: string };

/** Starts the background worker once, checks it answers, and reads the device. */
export function useIngestWorker(): WorkerState {
  const [state, setState] = useState<WorkerState>({ status: 'starting' });

  useEffect(() => {
    const { api, terminate } = createIngestWorker();
    let cancelled = false;

    (async () => {
      try {
        if ((await api.ping()) !== 'pong') throw new Error('Worker did not answer');
        const caps = await api.probeDevice();
        if (!cancelled) setState({ status: 'ready', caps, api });
      } catch (err) {
        if (!cancelled) setState({ status: 'error', message: errorMessage(err) });
      }
    })();

    return () => {
      cancelled = true;
      terminate();
    };
  }, []);

  return state;
}
