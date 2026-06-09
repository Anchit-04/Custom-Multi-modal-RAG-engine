import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from transformers import ASTModel

class UnifiedMultimodalEncoder(nn.Module):
    def __init__(self, shared_dim=512):
        super().__init__()
        # Initialize backbones
        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self.visual_backbone = clip_model.visual
        self.audio_backbone = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        
        # Custom Projection & Normalization Heads
        self.vis_proj = nn.Linear(512, shared_dim)
        self.aud_proj = nn.Linear(768, shared_dim)
        self.ln_vis = nn.LayerNorm(shared_dim)
        self.ln_aud = nn.LayerNorm(shared_dim)

    def encode_vision(self, visual_tensors):
        with torch.no_grad():
            x_v = self.visual_backbone(visual_tensors)
        z_v = self.ln_vis(self.vis_proj(x_v))
        return F.normalize(z_v, p=2, dim=-1)

    def encode_audio(self, spectrogram_tensors):
        with torch.no_grad():
            outputs = self.audio_backbone(spectrogram_tensors)
            x_a = outputs.last_hidden_state[:, 0, :] 
        z_a = self.ln_aud(self.aud_proj(x_a))
        return F.normalize(z_a, p=2, dim=-1)