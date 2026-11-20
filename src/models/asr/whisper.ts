/**
 * Whisper through transformers.js, with word-level timestamps.
 *
 * Needs a "_timestamped" ONNX export: word timings come from the decoder's
 * cross-attention, which the plain exports don't output.
 */
import { pipeline, type AutomaticSpeechRecognitionPipeline, type ProgressInfo } from '@huggingface/transformers';
import type { Word } from '../../core/types';
import type { InferenceDevice, SpeechProgress, WordTranscript } from './types';

/** One pipeline per model and device, created once and reused for every clip. */
const pipelines = new Map<string, Promise<AutomaticSpeechRecognitionPipeline>>();

function load(
  modelId: string,
  device: InferenceDevice,
  onProgress: (p: SpeechProgress) => void,
): Promise<AutomaticSpeechRecognitionPipeline> {
  const key = `${modelId}@${device}`;
  const existing = pipelines.get(key);
  if (existing) return existing;

  const created = pipeline('automatic-speech-recognition', modelId, {
    device,
    // WebGPU: full-precision encoder (accuracy), 4-bit decoder (speed).
    // CPU: 8-bit everywhere, the usual sweet spot for WASM.
    dtype: device === 'webgpu' ? { encoder_model: 'fp32', decoder_model_merged: 'q4' } : 'q8',
    progress_callback: (info: ProgressInfo) => {
      if (info.status === 'progress_total') {
        onProgress({ stage: 'loading-model', model: modelId, loaded: info.loaded, total: info.total });
      }
    },
  });
  pipelines.set(key, created);
  created.catch(() => pipelines.delete(key)); // let a later call retry after a failed load
  return created;
}

export async function transcribeWithWhisper(
  audio: Float32Array,
  modelId: string,
  device: InferenceDevice,
  onProgress: (p: SpeechProgress) => void,
): Promise<WordTranscript> {
  const asr = await load(modelId, device, onProgress);
  onProgress({ stage: 'transcribing', model: modelId, device });

  const output = await asr(audio, {
    return_timestamps: 'word',
    chunk_length_s: 30,
    stride_length_s: 5,
    language: 'en',
    task: 'transcribe',
  });
  const result = Array.isArray(output) ? output[0] : output;
  return { words: toWords(result?.chunks ?? []), model: modelId, device };
}

function toWords(chunks: Array<{ text: string; timestamp: [number, number | null] }>): Word[] {
  const words: Word[] = [];
  for (const chunk of chunks) {
    const text = chunk.text.trim();
    if (!text) continue;
    const [start, end] = chunk.timestamp;
    // The very last word can come back without an end time.
    words.push({ text, start, end: end ?? start + 0.3 });
  }
  return words;
}
