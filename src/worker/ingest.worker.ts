/**
 * Background worker: all model work runs here so the UI never freezes.
 *
 * Comlink exposes `api`, so the main thread calls these like normal async
 * functions. Jobs run one at a time per worker; the UI queues clips.
 */
import * as Comlink from 'comlink';
import { probeDevice } from '../models/device';
import type { SpeechProgress, SpeechResult } from '../models/asr/types';
import { runSpeech, type SpeechJob } from './tier1';

const api = {
  ping(): string {
    return 'pong';
  },

  probeDevice,

  /** Transcribes one clip. `onProgress` must be wrapped in Comlink.proxy() by the caller. */
  transcribe(job: SpeechJob, onProgress: (p: SpeechProgress) => void): Promise<SpeechResult> {
    return runSpeech(job, onProgress);
  },
};

export type IngestApi = typeof api;

Comlink.expose(api);
