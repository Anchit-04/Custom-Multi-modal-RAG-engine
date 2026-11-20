/**
 * Speech recognition entry point: picks the model for the profile and falls
 * back from WebGPU to CPU if the GPU path fails.
 */
import type { ModelChoice } from '../profiles';
import type { InferenceDevice, SpeechProgress, WordTranscript } from './types';
import { transcribeWithWhisper } from './whisper';

/** Used until a Parakeet runtime is wired in, and as the lite-profile model. */
export const WHISPER_BASE = 'onnx-community/whisper-base_timestamped';

export async function transcribeWords(
  audio: Float32Array,
  choice: ModelChoice,
  preferWebGPU: boolean,
  onProgress: (p: SpeechProgress) => void,
): Promise<WordTranscript> {
  // Parakeet needs its own ONNX runtime wrapper; until that lands, every
  // profile runs Whisper. The result names the model that actually ran.
  const modelId = choice.source.startsWith('onnx-community/whisper') ? choice.source : WHISPER_BASE;

  const devices: InferenceDevice[] = preferWebGPU ? ['webgpu', 'wasm'] : ['wasm'];
  let lastError: unknown;
  for (const device of devices) {
    try {
      return await transcribeWithWhisper(audio, modelId, device, onProgress);
    } catch (err) {
      lastError = err;
      console.warn(`[asr] ${modelId} failed on ${device}`, err);
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}
