import { useEffect, useRef, useState } from 'react';
import type { Seconds } from '../../core/types';

/** A request to jump to a time. `n` changes on every request so the same time can be requested twice. */
export interface SeekRequest {
  t: Seconds;
  n: number;
}

interface Props {
  file: File | null;
  seek: SeekRequest | null;
  onTime(t: Seconds): void;
}

/** Start a moment early so the jump lands just before the word, not mid-syllable. */
const LEAD_IN_SEC = 0.4;

export function Player({ file, seek, onTime }: Props) {
  const video = useRef<HTMLVideoElement>(null);
  const pendingSeek = useRef<Seconds | null>(null);
  const [url, setUrl] = useState<string | null>(null);

  // A blob URL lets the <video> element read the local file directly.
  useEffect(() => {
    if (!file) {
      setUrl(null);
      return;
    }
    const next = URL.createObjectURL(file);
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [file]);

  useEffect(() => {
    const el = video.current;
    if (!seek || !el) return;
    const t = Math.max(0, seek.t - LEAD_IN_SEC);
    if (el.readyState >= HTMLMediaElement.HAVE_METADATA) jumpTo(el, t);
    else pendingSeek.current = t; // the new file hasn't loaded yet; apply once it has
  }, [seek]);

  const onLoaded = () => {
    const el = video.current;
    if (el && pendingSeek.current !== null) {
      jumpTo(el, pendingSeek.current);
      pendingSeek.current = null;
    }
  };

  if (!url) {
    return <div className="player player-empty muted">Pick a clip or a search result to play it here.</div>;
  }

  return (
    <video
      ref={video}
      className="player"
      src={url}
      controls
      preload="metadata"
      onLoadedMetadata={onLoaded}
      onTimeUpdate={(e) => onTime(e.currentTarget.currentTime)}
    />
  );
}

function jumpTo(el: HTMLVideoElement, t: Seconds) {
  el.currentTime = t;
  void el.play().catch(() => {
    // Autoplay can be blocked until the user interacts with the page; the jump still happened.
  });
}
