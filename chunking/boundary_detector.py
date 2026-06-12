import torch
import torch.nn.functional as F
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Boundary:
    frame_idx: int
    timestamp: float
    confidence: float       # 0-1, higher = more confident it's a real cut
    source: str             # 'transcript', 'visual', 'both'

class BoundaryDetector:
    def __init__(self, cfg):
        self.cfg = cfg
    
    def detect(self,
               frame_embeds: torch.Tensor,    # (N, 512)
               frame_timestamps: List[float],
               whisper_result: dict) -> List[Boundary]:
        
        # --- Visual boundaries ---
        vis_dist = 1 - F.cosine_similarity(
            frame_embeds[:-1], frame_embeds[1:], dim=-1
        )
        vis_dist = self._smooth(vis_dist, window=4)
        
        # --- Transcript boundaries ---
        sentence_times = self._extract_sentence_ends(whisper_result)
        
        # --- Merge: confirm transcript boundary if visual also shifts ---
        boundaries = []
        for t in sentence_times:
            # Find closest frame to this timestamp
            frame_idx = min(
                range(len(frame_timestamps)),
                key=lambda i: abs(frame_timestamps[i] - t)
            )
            # Check visual distance in a ±5 frame window
            window_start = max(0, frame_idx - 5)
            window_end = min(len(vis_dist)-1, frame_idx + 5)
            local_max_dist = vis_dist[window_start:window_end].max().item()
            
            source = 'transcript'
            confidence = 0.5
            
            if local_max_dist > self.cfg.semantic_threshold:
                source = 'both'
                confidence = min(1.0, local_max_dist * 2)
            
            boundaries.append(Boundary(
                frame_idx=frame_idx,
                timestamp=t,
                confidence=confidence,
                source=source
            ))
        
        # Add strong visual-only boundaries 
        # (scene cuts with no speech — action sequences)
        vis_peaks = self._detect_peaks(vis_dist, self.cfg.semantic_threshold * 1.5)
        for idx in vis_peaks:
            # Only add if no transcript boundary is nearby (>2s away)
            t = frame_timestamps[idx]
            nearby = any(abs(b.timestamp - t) < 2.0 for b in boundaries)
            if not nearby:
                boundaries.append(Boundary(
                    frame_idx=idx,
                    timestamp=t,
                    confidence=0.7,
                    source='visual'
                ))
        
        boundaries.sort(key=lambda b: b.timestamp)
        return boundaries
    
    def _extract_sentence_ends(self, whisper_result) -> List[float]:
        sentence_enders = {'.', '!', '?'}
        times = []
        for segment in whisper_result.get('segments', []):
            text = segment.get('text', '').strip()
            if any(text.endswith(p) for p in sentence_enders):
                times.append(segment['end'])
        return times
    
    def _smooth(self, signal: torch.Tensor, window: int) -> torch.Tensor:
        kernel = torch.ones(window) / window
        padded = F.pad(signal.unsqueeze(0).unsqueeze(0),
                      (window//2, window//2), mode='replicate')
        return F.conv1d(padded, kernel.view(1,1,-1)).squeeze()
    
    def _detect_peaks(self, signal, threshold) -> List[int]:
        peaks = []
        s = signal.numpy()
        for i in range(1, len(s)-1):
            if s[i] > threshold and s[i] > s[i-1] and s[i] > s[i+1]:
                peaks.append(i)
        return peaks