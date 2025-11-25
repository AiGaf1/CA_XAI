import torch
import torch.nn as nn
import conf
from models.LTE import LTEOrig
import torch.nn.functional as F

def norm_embeddings(embeddings):
    # return embeddings / torch.sqrt((embeddings ** 2).sum(dim=-1, keepdims=True))
    return F.normalize(embeddings, dim=1, p=2)

def conv_prelu(in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.PReLU(out_channels),
    )


class Sum(nn.Module):
    """Apply several modules to the same input and sum the outputs."""
    def __init__(self, *modules: nn.Module):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum(m(x) for m in self.modules_list)


def residual_block(channels: int, pool: bool = False) -> nn.Sequential:
    """
    A reusable block:
        (conv + PReLU) + (AvgPool or Identity)
        summed with MaxPool in parallel.
    If pool=True, both branches downsample by 2.
    """
    pool_layer = nn.AvgPool1d(2) if pool else nn.Identity()
    max_pool = nn.MaxPool1d(2) if pool else nn.Identity()

    return nn.Sequential(
        Sum(
            nn.Sequential(
                conv_prelu(channels, channels),
                pool_layer,
            ),
            max_pool,
        ),
        nn.PReLU(channels),
    )


class LearnPeriodsKeyEmb(nn.Module):
    def __init__(self, periods_dict, output_size=512, hidden_size=128,
                 sequence_length=128, vocab_size=256, key_emb_dim=16):
        super().__init__()
        self.time_encoders = nn.ModuleDict()
        self.key_embedding = nn.Embedding(vocab_size, key_emb_dim)

        for feat, periods in periods_dict.items():
            self.time_encoders[feat] = LTEOrig(init_periods=periods)

        input_size = sum(encoder.d_out for encoder in self.time_encoders.values()) + key_emb_dim
        self.network = nn.Sequential(
            # Initial conv
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.PReLU(hidden_size),

            # Two pooled residual-style blocks at hidden_size
            residual_block(hidden_size, pool=True),
            residual_block(hidden_size, pool=True),

            # Channel expansion
            nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=3, padding=1),
            nn.PReLU(hidden_size * 2),

            # One pooled residual-style block at 2*hidden_size
            residual_block(hidden_size * 2, pool=True),

            # One non-pooled residual-style block (conv branch + identity branch)
            Sum(
                conv_prelu(hidden_size * 2, hidden_size * 2),
                nn.Identity(),
            ),
            nn.PReLU(hidden_size * 2),

            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size * sequence_length * 2 // 8, output_size),
        )

    def forward(self, x):
        hold = x[..., 0]
        flight = x[..., 1]
        keys = x[..., 2].long()

        encoded_x = [
            self.time_encoders["hold"](hold),
            self.time_encoders["flight"](flight),
            self.key_embedding(keys)  # key
        ]
        encoded_x = torch.cat(encoded_x, dim=-1)
        encoded_x = encoded_x.transpose(2, 1)
        embedding = self.network(encoded_x)
        return embedding
