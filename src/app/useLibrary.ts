/**
 * The clip library: what's been added, and the import queue that turns each
 * clip into a transcript, one clip at a time.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import * as Comlink from 'comlink';
import type { Clip, Utterance } from '../core/types';
import { config } from '../core/config';
import type { ProfileName } from '../models/device';
import type { SpeechProgress, SpeechResult } from '../models/asr/types';
import { clipIdFor, type PickedFile } from '../media/folder';
import { probeVideo } from '../media/probe';
import { decodeAudio } from '../media/decodeAudio';
import type { WorkerState } from './useIngestWorker';
import { errorMessage } from './format';

export type ItemStatus = 'queued' | 'decoding' | 'transcribing' | 'done' | 'error';

export interface LibraryItem {
  clip: Clip;
  file: File;
  status: ItemStatus;
  utterances: Utterance[];
  speech?: Omit<SpeechResult, 'utterances'>;
  error?: string;
}

export interface ModelDownload {
  model: string;
  loaded: number;
  total: number;
}

export interface Library {
  items: LibraryItem[];
  /** Set while a model is downloading (first run only; the browser caches it). */
  download: ModelDownload | null;
  addFiles(picked: PickedFile[]): Promise<void>;
}

export function useLibrary(worker: WorkerState, profile: ProfileName | null): Library {
  const [items, setItems] = useState<LibraryItem[]>([]);
  const [download, setDownload] = useState<ModelDownload | null>(null);
  const busy = useRef(false);

  const patch = useCallback((id: string, change: Partial<LibraryItem>) => {
    setItems((prev) => prev.map((item) => (item.clip.id === id ? { ...item, ...change } : item)));
  }, []);

  const addFiles = useCallback(async (picked: PickedFile[]) => {
    const added: LibraryItem[] = [];
    for (const p of picked) {
      const base = { id: clipIdFor(p), name: p.file.name, path: p.path, kind: 'video' as const, sizeBytes: p.file.size };
      try {
        const info = await probeVideo(p.file);
        added.push({ clip: { ...base, ...info }, file: p.file, status: 'queued', utterances: [] });
      } catch (err) {
        added.push({ clip: { ...base, durationSec: 0 }, file: p.file, status: 'error', utterances: [], error: errorMessage(err) });
      }
    }
    setItems((prev) => {
      const known = new Set(prev.map((item) => item.clip.id));
      return [...prev, ...added.filter((item) => !known.has(item.clip.id))];
    });
  }, []);

  // The queue: whenever nothing is running, start the next queued clip.
  useEffect(() => {
    if (worker.status !== 'ready' || !profile || busy.current) return;
    const next = items.find((item) => item.status === 'queued');
    if (!next) return;

    busy.current = true;
    const id = next.clip.id;
    const onProgress = (p: SpeechProgress) => {
      if (p.stage === 'loading-model') setDownload({ model: p.model, loaded: p.loaded, total: p.total });
      else setDownload(null);
    };

    void (async () => {
      let outcome: Partial<LibraryItem>;
      try {
        patch(id, { status: 'decoding' });
        const audio = await decodeAudio(next.file, config.ingest.audioSampleRate);
        patch(id, { status: 'transcribing' });
        const job = { clipId: id, audio, profile, webgpu: worker.caps.webgpu };
        // transfer: hand the audio buffer to the worker instead of copying it
        const { utterances, ...speech } = await worker.api.transcribe(
          Comlink.transfer(job, [audio.buffer]),
          Comlink.proxy(onProgress),
        );
        outcome = { status: 'done', utterances, speech };
      } catch (err) {
        outcome = { status: 'error', error: errorMessage(err) };
      }
      busy.current = false;
      setDownload(null);
      patch(id, outcome); // this state change re-runs the effect, which picks the next clip
    })();
  }, [items, worker, profile, patch]);

  return { items, download, addFiles };
}
