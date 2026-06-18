import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAggregator(nn.Module):
    """
    Aggregates a sequence of per-frame embeddings into a single chunk
    embedding using a lightweight 2-layer transformer encoder.

    Architecture:
      1. Add learnable temporal position encodings to each frame embed.
      2. Prepend a learnable [CLS] aggregation token.
      3. Run through N transformer encoder layers.
      4. Return the [CLS] output as the chunk representation.

    This is O(T²) in the number of frames T, but T ≤ 16 so it's negligible.
    """

    def __init__(
        self,
        shared_dim: int = 512,
        num_frames: int = 16,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.shared_dim = shared_dim
        self.num_frames = num_frames

        # Learnable temporal position encodings — one per frame slot
        self.temporal_pos = nn.Parameter(
            torch.randn(num_frames, shared_dim) * 0.02
        )

        # Learnable [CLS] aggregation token
        self.cls_token = nn.Parameter(torch.randn(1, 1, shared_dim) * 0.02)

        # Lightweight transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=shared_dim,
            nhead=num_heads,
            dim_feedforward=shared_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Optional: learnable video-level positional embedding for
        # chunk position within a full video (up to 10k chunks)
        self.chunk_pos_embed = nn.Embedding(10_000, shared_dim)

    def forward(
        self,
        frame_embeds: torch.Tensor,          # (B, T, shared_dim)
        chunk_positions: torch.Tensor = None, # (B,) int64 — position in video
    ) -> torch.Tensor:
        """
        Returns (B, shared_dim) L2-normalised chunk embedding.
        """
        B, T, D = frame_embeds.shape

        # Add temporal position encodings (broadcast over batch)
        pos = self.temporal_pos[:T].unsqueeze(0)   # (1, T, D)
        x = frame_embeds + pos                      # (B, T, D)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)     # (B, 1, D)
        x = torch.cat([cls, x], dim=1)             # (B, T+1, D)

        # Transformer encoding
        x = self.transformer(x)                    # (B, T+1, D)

        # Extract [CLS] output as chunk representation
        z = x[:, 0, :]                             # (B, D)

        # Add video-level chunk position if provided
        if chunk_positions is not None:
            z = z + self.chunk_pos_embed(chunk_positions)

        return F.normalize(z, p=2, dim=-1)