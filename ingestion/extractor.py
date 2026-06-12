import subprocess
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import librosa

@dataclass
class RawVideo:
    frames: List[np.ndarray]   # (H, W, 3) uint8, RGB
    frame_timestamps: List[float]
    audio: np.ndarray          # (samples,) float32 mono
    fps: float
    duration: float

def extract(video_path: str, target_fps: int = 2) -> RawVideo:
    path = Path(video_path)
    audio_path = path.with_suffix('.wav')
    
    # Extract audio via ffmpeg
    subprocess.run([
        'ffmpeg', '-y', '-i', str(path),
        '-ac', '1', '-ar', '16000',
        str(audio_path)
    ], capture_output=True, check=True)
    
    audio, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    audio_path.unlink()  # clean up
    
    # Extract frames
    cap = cv2.VideoCapture(str(path))
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / native_fps
    
    frame_interval = int(native_fps / target_fps)
    frames, timestamps = [], []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            timestamps.append(frame_idx / native_fps)
        frame_idx += 1
    
    cap.release()
    return RawVideo(frames, timestamps, audio, target_fps, duration)