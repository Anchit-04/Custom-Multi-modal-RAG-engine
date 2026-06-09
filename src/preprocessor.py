import os
import cv2
import numpy as np
import torch
import torchaudio.transforms as T
import librosa

class SemanticPreprocessor:
    def __init__(self, target_sr=16000, fps=1, diff_threshold=25.0):
        self.target_sr = target_sr
        self.fps = fps
        self.diff_threshold = diff_threshold # Percentage of pixels changed
        self.mel_transform = T.MelSpectrogram(sample_rate=target_sr, n_mels=128)

    def process_video(self, video_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        y, _ = librosa.load(video_path, sr=self.target_sr, mono=True)
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames, prev_gray = [], None
        chunk_idx, start_frame = 0, 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Check for Semantic Scene Cut
            cut_detected = False
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                changed_pixels = np.count_nonzero(diff > 30)
                if (changed_pixels / diff.size) * 100 > self.diff_threshold:
                    cut_detected = True

            # We only sample frames based on our target FPS
            if current_frame % int(video_fps / self.fps) == 0:
                resized = cv2.resize(frame, (224, 224))
                frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

            if cut_detected and len(frames) > 0:
                self._export_chunk(y, frames, start_frame, current_frame, video_fps, video_name, chunk_idx, output_dir)
                frames = []
                start_frame = current_frame
                chunk_idx += 1
                
            prev_gray = gray
        
        cap.release()
        print(f"[{video_name}] Extracted {chunk_idx} semantic chunks.")

    def _export_chunk(self, full_audio, frames, start_f, end_f, v_fps, name, c_idx, out_dir):
        start_sec, end_sec = start_f / v_fps, end_f / v_fps
        audio_chunk = full_audio[int(start_sec * self.target_sr) : int(end_sec * self.target_sr)]
        
        # Ensure minimum audio length for AST compatibility
        if len(audio_chunk) < self.target_sr:
            audio_chunk = np.pad(audio_chunk, (0, self.target_sr - len(audio_chunk)))
            
        mel_spec = self.mel_transform(torch.tensor(audio_chunk).float().unsqueeze(0))
        log_mel = torch.log(mel_spec + 1e-6).squeeze(0).T
        
        v_tensor = torch.tensor(np.array(frames)).permute(0, 3, 1, 2).float() / 255.0
        
        torch.save({
            'chunk_id': f"{name}_{c_idx:04d}",
            'timestamps': (start_sec, end_sec),
            'video_tensor': v_tensor,
            'audio_spectrogram': log_mel
        }, os.path.join(out_dir, f"{name}_{c_idx:04d}.pt"))