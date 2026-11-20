import * as Comlink from 'comlink';
import type { IngestApi } from './ingest.worker';

export interface IngestWorker {
  api: Comlink.Remote<IngestApi>;
  terminate(): void;
}

export function createIngestWorker(): IngestWorker {
  const worker = new Worker(new URL('./ingest.worker.ts', import.meta.url), { type: 'module' });
  return {
    api: Comlink.wrap<IngestApi>(worker),
    terminate: () => worker.terminate(),
  };
}
