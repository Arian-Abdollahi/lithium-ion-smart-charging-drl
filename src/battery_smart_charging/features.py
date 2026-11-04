"""Original Set Transformer extractor from the uploaded attention-training notebook."""

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SetTransformerExtractor(BaseFeaturesExtractor):
    """
    Self-Attention + Learned Pooling
    Attention واقعاً تصمیم می‌گیرد کدام سلول مهم‌تر است
    """

    def __init__(
        self,
        observation_space,
        n_cells: int,
        cell_feat_dim: int = 3,
        d_model: int = 32,
        n_heads: int = 4,
        latent_dim: int = 64,
    ):
        super().__init__(observation_space, features_dim=latent_dim)

        obs_dim = observation_space.shape[0]
        if obs_dim != n_cells * cell_feat_dim:
            raise ValueError("Observation dimension must equal n_cells * cell_feat_dim")

        self.n_cells = n_cells
        self.cell_feat_dim = cell_feat_dim

        # 1) shared cell embedding
        self.cell_embed = nn.Sequential(
            nn.Linear(cell_feat_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        # 2) self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=False,
        )

        # 3) learned pooling (THIS is the missing piece)
        self.pool = nn.Linear(d_model, 1)

        # 4) final projection
        self.post = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.ReLU(),
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        B = obs.shape[0]

        # (B, 3N) -> (B, N, 3)
        x = obs.view(B, self.n_cells, self.cell_feat_dim)

        # (B, N, d_model)
        e = self.cell_embed(x)

        # attention expects (N, B, d_model)
        e_t = e.transpose(0, 1)

        # self-attention
        h_t, _ = self.attn(e_t, e_t, e_t, need_weights=False)

        # back to (B, N, d_model)
        h = h_t.transpose(0, 1)

        # -------------------------
        # LEARNED POOLING (KEY PART)
        # -------------------------
        # (B, N, 1) -> (B, N)
        scores = self.pool(h).squeeze(-1)

        # attention weights over cells
        weights = th.softmax(scores, dim=1)

        # weighted sum
        pooled = (h * weights.unsqueeze(-1)).sum(dim=1)

        # final latent
        z = self.post(pooled)
        return z
