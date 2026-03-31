import torch
import torch.nn as nn
from models.fourier_encoding import LearnableFourierFeatures
from utils.metrics import norm_embeddings

# Number of raw timing features (hold time + flight time) when not using MSTE
RAW_TIME_FEATURES = 2


def conv_prelu(in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1) -> nn.Sequential:
    """Conv1d → BatchNorm → PReLU block."""
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.PReLU(out_channels),
    )


class ResidualBlock(nn.Module):
    """Conv block with a skip connection. Optionally halves the sequence length via AvgPool."""

    def __init__(self, channels: int, downsample: bool = False):
        super().__init__()
        pool = nn.AvgPool1d(2) if downsample else nn.Identity()
        self.conv_path = nn.Sequential(conv_prelu(channels, channels), pool)
        self.skip_path = nn.AvgPool1d(2) if downsample else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_path(x) + self.skip_path(x)


class KeystrokeCNN(nn.Module):
    """
    CNN encoder for keystroke sequences.

    Input tensor shape: (B, L, 3)  — columns: [hold_time, flight_time, key_id]
    Output shape:       (B, output_size)  — L2-normalised embedding
    """

    def __init__(
        self,
        periods_dict,
        output_size: int = 256,
        hidden_size: int = 128,
        vocab_size: int = 256,
        key_emb_dim: int = 16,
        n_periods: int = 16,
        use_mste: bool = True,
    ):
        super().__init__()
        self.use_mste = use_mste

        # --- Feature encoders ---
        self.key_embedding = nn.Embedding(vocab_size, key_emb_dim)

        if use_mste:
            self.time_encoder = LearnableFourierFeatures(periods_dict, num_features=n_periods)
            time_feat_dim = self.time_encoder.d_out
        else:
            self.time_encoder = None
            time_feat_dim = RAW_TIME_FEATURES

        input_size = time_feat_dim + key_emb_dim

        # --- CNN backbone (3 downsampling stages: L → L/2 → L/4 → L/8) ---
        wide = hidden_size * 2

        self.backbone = nn.Sequential(
            conv_prelu(input_size, hidden_size),
            ResidualBlock(hidden_size, downsample=True),
            ResidualBlock(hidden_size, downsample=True),
            conv_prelu(hidden_size, wide),
            ResidualBlock(wide, downsample=True),
            ResidualBlock(wide, downsample=False),
        )

        # --- Projection head ---
        self.projector = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(wide, output_size, bias=False),
        )
        self.embedding_dim = output_size

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x.float()

        # Split input columns
        time_vec = x[..., :2]           # (B, L, 2)  hold + flight times
        keys     = x[..., 2].long()     # (B, L)     key ids

        # Encode each modality
        time_feat = self.time_encoder(time_vec) if self.use_mste else time_vec
        key_feat  = self.key_embedding(keys)

        # Concatenate and pass through CNN (channels-first)
        features = torch.cat([time_feat, key_feat], dim=-1)   # (B, L, C)
        features = self.backbone(features.transpose(1, 2))     # (B, wide, L')
        features = features.transpose(1, 2)                    # (B, L', wide)

        # Downsample mask by 3x AvgPool1d(2) to match backbone stride
        mask_ds = mask.float().unsqueeze(1)                    # (B, 1, L)
        for _ in range(3):
            mask_ds = torch.nn.functional.avg_pool1d(mask_ds, kernel_size=2, stride=2)
        valid = (mask_ds > 0).float().transpose(1, 2)          # (B, L', 1)

        embedding = (features * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-8)
        embedding = self.projector(embedding)
        return norm_embeddings(embedding)

    def get_embedding_dim(self) -> int:
        return self.embedding_dim