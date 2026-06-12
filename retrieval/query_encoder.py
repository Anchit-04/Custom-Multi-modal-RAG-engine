import numpy as np
import torch
import open_clip
from typing import Union


class QueryEncoder:
    """
    Encodes free-text queries into the shared 512-d embedding space
    using the CLIP ViT-B/32 text encoder + the same projection head
    that was used during video indexing.

    If a trained encoder checkpoint is available, load it to use the
    fine-tuned projection head. Otherwise falls back to vanilla CLIP text.
    """

    def __init__(self, encoder=None, device: str = "cpu"):
        """
        encoder: an instance of UnifiedMultimodalEncoder (optional).
                 If None, uses raw CLIP text embeddings (no projection).
        device:  'cpu' or 'cuda'
        """
        self.device = device

        if encoder is not None:
            self.encoder = encoder.to(device).eval()
            self._use_full_encoder = True
        else:
            # Lightweight fallback: just CLIP text, no projection
            clip_model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            self.clip_model = clip_model.to(device).eval()
            self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
            self._use_full_encoder = False

    def encode(self, query: str) -> np.ndarray:
        """
        Encode a text query string into a (512,) float32 numpy vector.
        """
        if self._use_full_encoder:
            return self._encode_with_encoder(query)
        return self._encode_clip_only(query)

    def _encode_with_encoder(self, query: str) -> np.ndarray:
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        tokens = tokenizer([query]).to(self.device)
        with torch.no_grad():
            z = self.encoder.encode_text(tokens)
        return z.squeeze(0).cpu().numpy().astype(np.float32)

    def _encode_clip_only(self, query: str) -> np.ndarray:
        tokens = self.tokenizer([query]).to(self.device)
        with torch.no_grad():
            z = self.clip_model.encode_text(tokens)
            z = torch.nn.functional.normalize(z, p=2, dim=-1)
        return z.squeeze(0).cpu().numpy().astype(np.float32)