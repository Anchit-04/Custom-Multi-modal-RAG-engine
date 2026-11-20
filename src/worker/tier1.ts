/**
 * Tier 1: everything that runs at import so a clip becomes searchable.
 * M1 covers speech; M2 adds frames and cuts, M3 adds screen text.
 */
import type { ClipId } from '../core/types';
import { config } from '../core/config';
import type { ProfileName } from '../models/device';
import { MODEL_PLANS } from '../models/profiles';
import { transcribeWords } from '../models/asr';
import { groupIntoUtterances } from '../models/asr/utterances';
import type { SpeechProgress, SpeechResult } from '../models/asr/types';

export interface SpeechJob {
  clipId: ClipId;
  /** Mono samples at config.ingest.audioSampleRate. */
  audio: Float32Array;
  profile: ProfileName;
  webgpu: boolean;
}

export async function runSpeech(job: SpeechJob, onProgress: (p: SpeechProgress) => void): Promise<SpeechResult> {
  let startedAt = 0;
  const transcript = await transcribeWords(job.audio, MODEL_PLANS[job.profile].speech, job.webgpu, (p) => {
    if (p.stage === 'transcribing') startedAt = performance.now();
    onProgress(p);
  });

  return {
    utterances: groupIntoUtterances(job.clipId, transcript.words),
    model: transcript.model,
    device: transcript.device,
    audioSec: job.audio.length / config.ingest.audioSampleRate,
    elapsedSec: (performance.now() - startedAt) / 1000,
  };
}
