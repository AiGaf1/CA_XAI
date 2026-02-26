import torch
import torch.nn as nn
from models.LTE import LearnableFourierFeatures
import torch.nn.functional as F
from collections import OrderedDict


def norm_embeddings(embeddings):
    # return embeddings / torch.sqrt((embeddings ** 2).sum(dim=-1, keepdims=True))
    return F.normalize(embeddings, dim=-1, p=2)


def conv_prelu(in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)),
        ('batch_norm', nn.BatchNorm1d(out_channels)),
        ('PReLU', nn.PReLU(out_channels))
    ]))


class ParallelSum(nn.Module):
    """Apply several modules to the same input and sum the outputs."""

    def __init__(self, *modules: nn.Module):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)
        # Learnable weights for each path to improve gradient flow

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize weights to sum to 1

        return sum(m(x) for m in self.modules_list)


def residual_block(channels: int, downsample: bool = False) -> nn.Sequential:
    """
    A reusable block:
        (conv + PReLU) + (AvgPool or Identity)
        summed with MaxPool in parallel.
    If pool=True, both branches downsample by 2.
    """
    pool_layer = nn.AvgPool1d(2) if downsample else nn.Identity()
    skip_layer = nn.MaxPool1d(2) if downsample else nn.Identity()

    return nn.Sequential(
        ParallelSum(
            nn.Sequential(conv_prelu(channels, channels), pool_layer, ),
            skip_layer
        ),
        nn.PReLU(channels)
    )


class CNN_LTE(nn.Module):
    def __init__(self, periods_dict, output_size=512, hidden_size=128,
                 sequence_length=128, vocab_size=256, key_emb_dim=16, use_projector=False,
                 n_periods=16):
        super().__init__()
        self.use_projector = use_projector
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.time_encoders = LearnableFourierFeatures(periods_dict, num_features=n_periods)
        self.key_embedding = nn.Embedding(vocab_size, key_emb_dim)

        # Updated: Simplified input_size calculation
        input_size = self.time_encoders.d_out + key_emb_dim

        # Calculate flattened dimension after downsampling (3 pooling layers = 2^3 = 8x reduction)
        # flat_dim = (hidden_size * 2) * (sequence_length // 16)
        flat_dim = hidden_size * 2

        self.backbone = nn.Sequential(OrderedDict([
            # Initial conv
            ("enc1_conv", conv_prelu(input_size, hidden_size)),
            # Downsampling sequence_length by residual blocks (128 -> 64 -> 32 -> 16 -> 8)
            ("res1_down", residual_block(hidden_size, downsample=True)),
            ("res2_down", residual_block(hidden_size, downsample=True)),
            # Channel expansion
            ("enc2_conv", conv_prelu(hidden_size, hidden_size * 2)),
            # One pooled residual-style block at 2*hidden_size
            ("res3_down", residual_block(hidden_size * 2, downsample=True)),

            # Deep feature extraction
            ("res4", residual_block(hidden_size * 2, downsample=True)),
            ("res5", residual_block(hidden_size * 2, downsample=False)),

            ("flatten", nn.Flatten())
        ]))

        if self.use_projector:
            self.projector = nn.Sequential(OrderedDict([
                ("bn_flat", nn.BatchNorm1d(flat_dim)),
                ("drop", nn.Dropout(p=0.2)),
                ("fc_out", nn.Linear(flat_dim, output_size, bias=False))
            ]))

        # Commented out to match Transformer pattern
        # self._init_weights()

    def _init_weights(self):
        """Proper weight initialization for better gradient flow"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.25)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.25)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask):
        """
        x: (B, L, 3) -> hold, flight, key
        mask: (B, L) -> 1 for valid, 0 for padding
        """
        x, mask = x.float(), mask.float()
        hold, flight, keys = x.unbind(dim=-1)
        keys = keys.long()

        # ----------------------------
        # 1. Fourier encoding (Updated to match Transformer)
        # ----------------------------
        time_vec = torch.stack([hold, flight], dim=-1)  # (B, L, 2)
        time_feat = self.time_encoders(time_vec)  # (B, L, 2D)

        # ----------------------------
        # 2. Key embedding
        # ----------------------------
        key_feat = self.key_embedding(keys)  # (B, L, K)

        # ----------------------------
        # 3. Combine features
        # ----------------------------
        encoded_x = torch.cat([time_feat, key_feat], dim=-1)  # (B, L, C)

        # Apply mask here to zero padded positions BEFORE the backbone
        # encoded_x = encoded_x * mask.unsqueeze(-1)

        # ----------------------------
        # 4. CNN Backbone
        # ----------------------------
        encoded_x = encoded_x.transpose(1, 2)  # (B, C, L)
        features = self.backbone(encoded_x)  # (B, flat_dim)

        # Using avg_pool to get proportional weights
        # ds_factor = encoded_x.shape[-1] // features.shape[-1]
        # mask = mask.unsqueeze(1)  # (B, 1, L) for pooling
        # mask_ds = F.avg_pool1d(mask, kernel_size=ds_factor, stride=ds_factor, padding=0).squeeze(1)  # (B, L')

        # Apply downsampled mask (fractional weights)
        # features = features * mask_ds.unsqueeze(1)  # (B, C', L')

        # Weighted average: sum over L', divide by sum of weights
        # denom = mask_ds.sum(dim=-1, keepdim=True) + 1e-8  # Avoid div by zero, (B, 1)
        # features = features.sum(dim=-1) / denom  # (B, C')

        # ----------------------------
        # 5. Optional projection
        # ----------------------------
        embedding = features
        if self.use_projector:
            embedding = self.projector(features)

        # ----------------------------
        # 6. Normalize
        # ----------------------------
        return norm_embeddings(embedding)

    def get_embedding_dim(self) -> int:
        """Return the dimension of output embeddings."""
        return self.output_size if self.use_projector else self.hidden_size * 2
