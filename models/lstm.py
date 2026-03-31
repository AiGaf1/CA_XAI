import torch
import torch.nn as nn
from models.fourier_encoding import LearnableFourierFeatures
from utils.metrics import norm_embeddings


class KeystrokeLSTM(nn.Module):
    def __init__(self, periods_dict, output_size=256, hidden_size=128,
                 vocab_size=256, key_emb_dim=16,
                 n_periods=16, use_mste=True, num_layers=2, dropout=0.1):
        super().__init__()
        self.use_mste = use_mste

        self.key_embedding = nn.Embedding(vocab_size, key_emb_dim)
        if use_mste:
            self.time_encoders = LearnableFourierFeatures(periods_dict, num_features=n_periods)
            input_size = self.time_encoders.d_out + key_emb_dim
        else:
            input_size = 2 + key_emb_dim  # raw hold + flight

        self.input_norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        lstm_out_dim = hidden_size * 2  # bidirectional

        self.projector = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(lstm_out_dim, output_size, bias=False),
        )
        self.embedding_dim = output_size

    def forward(self, x, mask):
        x, mask = x.float(), mask.float()
        keys = x[..., 2].long()
        time_vec = x[..., :2]
        time_feat = self.time_encoders(time_vec) if self.use_mste else time_vec
        key_feat = self.key_embedding(keys)

        encoded_x = torch.cat([time_feat, key_feat], dim=-1)  # (B, L, C)
        encoded_x = self.input_norm(encoded_x)

        out, _ = self.lstm(encoded_x)  # (B, L, 2*H)

        # Masked mean pooling
        valid = mask.unsqueeze(-1)
        embedding = (out * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-8)
        projector = self.projector(embedding)
        return norm_embeddings(projector)

    def get_embedding_dim(self) -> int:
        return self.embedding_dim
