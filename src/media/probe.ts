/** Reads a video's duration and size by letting the browser parse its header. */
import type { Seconds } from '../core/types';

export interface VideoInfo {
  durationSec: Seconds;
  width: number;
  height: number;
}

export function probeVideo(file: File): Promise<VideoInfo> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const video = document.createElement('video');
    const cleanup = () => {
      video.removeAttribute('src');
      video.load();
      URL.revokeObjectURL(url);
    };

    video.preload = 'metadata';
    video.muted = true;
    video.onloadedmetadata = () => {
      const info = { durationSec: video.duration, width: video.videoWidth, height: video.videoHeight };
      cleanup();
      if (!Number.isFinite(info.durationSec)) reject(new Error(`Could not read the length of ${file.name}`));
      else resolve(info);
    };
    video.onerror = () => {
      cleanup();
      reject(new Error(`This browser can't play ${file.name} (unsupported format or codec)`));
    };
    video.src = url;
  });
}
