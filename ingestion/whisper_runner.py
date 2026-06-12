import whisper
import gc
from typing import Dict, Any


def transcribe(video_path: str, model_size: str = "small") -> Dict[str, Any]:
    """
    Run Whisper on a video file and return the full result dict including
    word-level timestamps. The model is deleted and VRAM freed after use.

    Returns a dict with keys:
        text       — full transcript string
        segments   — list of segment dicts, each with:
                        id, start, end, text
        words      — flat list of word dicts:
                        word, start, end, probability
    """
    model = whisper.load_model(model_size)

    result = model.transcribe(
        video_path,
        word_timestamps=True,
        verbose=False,
    )

    # Flatten word timestamps from segments into top-level 'words' list
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append(
                {
                    "word": w["word"].strip(),
                    "start": w["start"],
                    "end": w["end"],
                    "probability": w.get("probability", 1.0),
                }
            )
    result["words"] = words

    # Explicitly free model so VRAM is available for encoders
    del model
    gc.collect()

    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return result