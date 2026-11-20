/**
 * Background worker: all model work runs here so the UI never freezes.
 *
 * Comlink exposes `api` so the main thread calls these like normal async
 * functions. M1 adds speech; later milestones add the tier 1 and tier 2
 * pipelines and the job scheduler.
 */
import * as Comlink from 'comlink';
import { probeDevice } from '../models/device';

const api = {
  ping(): string {
    return 'pong';
  },
  probeDevice,
};

export type IngestApi = typeof api;

Comlink.expose(api);
