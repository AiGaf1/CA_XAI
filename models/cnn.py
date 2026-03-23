import torch
import torch.nn as nn
from models.fourier_encoding import LearnableFourierFeatures
import torch.nn.functional as F
from collections import OrderedDict


def norm_embeddings(embeddings):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum(m(x) for m in self.modules_list)


def residual_block(channels: int, downsample: bool = False) -> nn.Sequential:
    pool_layer = nn.AvgPool1d(2) if downsample else nn.Identity()
    skip_layer = nn.AvgPool1d(2) if downsample else nn.Identity()

    return nn.Sequential(
        ParallelSum(
            nn.Sequential(conv_prelu(channels, channels), pool_layer),
            skip_layer
        ),
        nn.PReLU(channels)
    )


class KeystrokeCNN(nn.Module):
    def __init__(self, periods_dict, output_size=512, hidden_size=128,
                 sequence_length=128, vocab_size=256, key_emb_dim=16, use_projector=False,
                 n_periods=16, use_mste=True):
        super().__init__()
        self.use_projector = use_projector
        self.use_mste = use_mste

        self.key_embedding = nn.Embedding(vocab_size, key_emb_dim)
        if use_mste:
            self.time_encoders = LearnableFourierFeatures(periods_dict, num_features=n_periods)
            input_size = self.time_encoders.d_out + key_emb_dim
        else:
            input_size = 2 + key_emb_dim  # raw hold + flight

        # 3 downsamples: 50 → 25 → 12 → 6
        self.flat_dim = hidden_size * 2 * (sequence_length // 2 // 2 // 2)

        self.backbone = nn.Sequential(OrderedDict([
            ("enc1_conv", conv_prelu(input_size, hidden_size)),
            ("res1_down", residual_block(hidden_size, downsample=True)),
            ("res2_down", residual_block(hidden_size, downsample=True)),
            ("enc2_conv", conv_prelu(hidden_size, hidden_size * 2)),
            ("res3_down", residual_block(hidden_size * 2, downsample=True)),
            ("res4",      residual_block(hidden_size * 2, downsample=False)),
            ("flatten",   nn.Flatten())
        ]))

        if use_projector:
            self.projector = nn.Sequential(OrderedDict([
                ("drop",    nn.Dropout(p=0.1)),
                ("fc_out",  nn.Linear(self.flat_dim, output_size, bias=False))
            ]))
        self.embedding_dim = output_size if use_projector else self.flat_dim

    def forward(self, x, mask=None):
        x = x.float()
        hold, flight, keys = x.unbind(dim=-1)
        keys = keys.long()

        time_vec  = torch.stack([hold, flight], dim=-1)                                    # (B, L, 2)
        time_feat = self.time_encoders(time_vec) if self.use_mste else time_vec            # (B, L, 2D) or (B, L, 2)
        key_feat  = self.key_embedding(keys)                                  # (B, L, K)

        encoded_x = torch.cat([time_feat, key_feat], dim=-1)  # (B, L, C)
        features  = self.backbone(encoded_x.transpose(1, 2))  # (B, flat_dim)

        embedding = self.projector(features) if self.use_projector else features
        return norm_embeddings(embedding)

    def get_embedding_dim(self) -> int:
        return self.embedding_dim
