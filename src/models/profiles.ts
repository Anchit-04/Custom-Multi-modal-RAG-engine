/**
 * Which model each profile uses for each job.
 *
 * Repo ids point at ONNX exports on the Hugging Face Hub. Each one gets
 * confirmed (loads, runs, acceptable speed) in the milestone that first uses
 * it; until then treat them as the plan, not a guarantee.
 */
import type { ProfileName } from './device';

export interface ModelChoice {
  /** Short name for the UI. */
  label: string;
  /** Hugging Face repo id, or the npm package for non-HF runtimes. */
  source: string;
  /** Milestone that wires it in. */
  milestone: string;
}

export interface ModelPlan {
  speech: ModelChoice;
  visual: ModelChoice;
  /** Precise vectors for re-ranking; null = skip re-ranking. */
  rerank: ModelChoice | null;
  ocr: ModelChoice;
  /** Captions; null = off. */
  captions: ModelChoice | null;
  /** Meaning-based text search over speech and captions; null = keyword only. */
  textEmbed: ModelChoice | null;
}

const whisperBase: ModelChoice = {
  label: 'Whisper base (word timestamps)',
  source: 'onnx-community/whisper-base_timestamped',
  milestone: 'M1',
};
const parakeet: ModelChoice = {
  label: 'Parakeet TDT 0.6B v3',
  source: 'istupakov/parakeet-tdt-0.6b-v3-onnx',
  milestone: 'M1',
};
const mobileclip = (variant: 's0' | 's2' | 'b'): ModelChoice => ({
  label: `MobileCLIP2-${variant.toUpperCase()}`,
  source: `plhery/mobileclip2-onnx#${variant}`,
  milestone: 'M2',
});
const siglip = (size: 'base' | 'large'): ModelChoice => ({
  label: `SigLIP 2 ${size}`,
  source: `onnx-community/siglip2-${size}-patch16-256-ONNX`,
  milestone: 'M5',
});
const ppocr: ModelChoice = { label: 'PP-OCRv5', source: '@paddleocr/paddleocr-js', milestone: 'M3' };
const florence: ModelChoice = {
  label: 'Florence-2 base',
  source: 'onnx-community/Florence-2-base-ft',
  milestone: 'M5',
};
const embeddingGemma: ModelChoice = {
  label: 'EmbeddingGemma 300M (q8)',
  source: 'onnx-community/embeddinggemma-300m-ONNX',
  milestone: 'M5',
};

export const MODEL_PLANS: Record<ProfileName, ModelPlan> = {
  lite: {
    speech: whisperBase,
    visual: mobileclip('s0'),
    rerank: null,
    ocr: ppocr,
    captions: null,
    textEmbed: null,
  },
  standard: {
    speech: parakeet,
    visual: mobileclip('s2'),
    rerank: siglip('base'),
    ocr: ppocr,
    captions: florence,
    textEmbed: embeddingGemma,
  },
  max: {
    speech: parakeet,
    visual: mobileclip('b'),
    rerank: siglip('large'),
    ocr: ppocr,
    captions: florence,
    textEmbed: embeddingGemma,
  },
};
