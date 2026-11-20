/**
 * The shared data contract for framegrep.
 *
 * Everything in src/core works on these plain types, so it can be tested in Node
 * without a browser, a GPU or any models. Times are always seconds from the start
 * of a clip.
 */

export type ClipId = string;
export type Seconds = number;

/** A media file in the user's library. */
export interface Clip {
  id: ClipId;
  /** File name shown in the UI. */
  name: string;
  /** Path relative to the folder the user picked. */
  path: string;
  kind: 'video' | 'image';
  sizeBytes: number;
  /** 0 for images. */
  durationSec: Seconds;
  width?: number;
  height?: number;
}

/** One recognised word and its timing, from the speech model. */
export interface Word {
  text: string;
  start: Seconds;
  end: Seconds;
  confidence?: number;
}

/** A sentence-like stretch of speech: the unit the speech model returns. */
export interface Utterance {
  clipId: ClipId;
  text: string;
  start: Seconds;
  end: Seconds;
  words: Word[];
}

/** A frame sampled from a clip, with its L2-normalised visual embedding. */
export interface FrameEmbedding {
  clipId: ClipId;
  t: Seconds;
  vector: Float32Array;
}

/** A detected cut point and the evidence for it (port of the Python Boundary dataclass). */
export interface Boundary {
  t: Seconds;
  frameIndex: number;
  /** 0 to 1; higher means more likely a real cut. */
  confidence: number;
  source: 'transcript' | 'visual' | 'both';
}

/**
 * A searchable stretch of a clip. Micro segments come from the segmenter;
 * meso segments group several consecutive micro segments.
 */
export interface Segment {
  id: string;
  clipId: ClipId;
  level: 'micro' | 'meso';
  start: Seconds;
  end: Seconds;
  /** Representative frame time, used for thumbnails, OCR and re-ranking vectors. */
  keyframeT: Seconds;
  /** Speech inside this segment, joined. Empty if nobody speaks. */
  transcript: string;
}

/** Text read off the screen at one keyframe (OCR). */
export interface ScreenText {
  clipId: ClipId;
  t: Seconds;
  text: string;
}

/** Which index a match came from. */
export type Signal = 'speech' | 'screen' | 'visual' | 'caption';

/** One match from one index. */
export interface Hit {
  clipId: ClipId;
  start: Seconds;
  end: Seconds;
  /** Higher is better. Only comparable within one index; fusion uses ranks. */
  score: number;
  signal: Signal;
  /** Id of the indexed document (utterance, OCR frame, segment) that matched. */
  docId: string;
  /** The text that matched, for the "why it matched" chips in the UI. */
  matched?: string;
}

/** A final result: one time span plus every hit that supports it. */
export interface FusedHit {
  clipId: ClipId;
  start: Seconds;
  end: Seconds;
  score: number;
  hits: Hit[];
}

/**
 * Query fields:
 *   say:    spoken words (speech index)
 *   screen: on-screen text (OCR index)
 *   shows:  what's visible (visual vectors, plus captions when available)
 *   any:    no field given; the planner searches everything
 */
export type QueryField = 'say' | 'screen' | 'shows' | 'any';

/** Parsed query tree, produced by core/query/parser.ts. */
export type QueryNode =
  | { kind: 'term'; field: QueryField; text: string; phrase: boolean }
  | { kind: 'and'; children: QueryNode[] }
  | { kind: 'or'; children: QueryNode[] }
  | { kind: 'not'; child: QueryNode }
  | { kind: 'near'; withinSec: Seconds; children: QueryNode[] };
