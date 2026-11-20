import type { Seconds, Utterance, Word } from '../../core/types';

export type InferenceDevice = 'webgpu' | 'wasm';

/** Progress reported back to the UI while a clip is being transcribed. */
export type SpeechProgress =
  | { stage: 'loading-model'; model: string; loaded: number; total: number }
  | { stage: 'transcribing'; model: string; device: InferenceDevice };

/** Raw output of a speech model: timed words, before grouping into utterances. */
export interface WordTranscript {
  words: Word[];
  model: string;
  device: InferenceDevice;
}

/** What the worker returns for one clip. */
export interface SpeechResult {
  utterances: Utterance[];
  model: string;
  device: InferenceDevice;
  audioSec: Seconds;
  /** Wall-clock time spent transcribing, excluding the first model download. */
  elapsedSec: Seconds;
}
