import torch
import torch.nn as nn
import conf
from models.LTE import LearnableFourierFeatures
import torch.nn.functional as F
from collections import OrderedDict

def norm_embeddings(embeddings):
    return F.normalize(embeddings, dim=-1, p=2)


class Transformer_LTE(nn.Module):
    def __init__(self, periods_dict, output_size=64, hidden_size=128,
                 window_size=50, vocab_size=256, key_emb_dim=16, use_projector=True,
                 num_layers=4, num_heads=2, ff_dim=512, dropout=0.2):
        super().__init__()
        self.key_embedding = nn.Embedding(vocab_size, key_emb_dim)
        self.use_projector = use_projector
        self.window_size = window_size
        self.d_model = hidden_size
        self.time_encoders = LearnableFourierFeatures(periods_dict, num_features=conf.N_PERIODS)
        input_size = self.time_encoders.d_out + key_emb_dim

        self.input_proj = nn.Linear(input_size, self.d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, window_size, self.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if self.use_projector:
            self.projector = nn.Sequential(OrderedDict([
                # ("bn", nn.BatchNorm1d(self.d_model)),
                ("drop", nn.Dropout(p=0.1)),
                ("fc_out", nn.Linear(self.d_model, output_size, bias=False))
            ]))

    def forward(self, x, mask):
        x, mask = x.float(), mask.float()
        hold, flight, keys = x.unbind(dim=-1)
        keys = keys.long()

        # 1. Fourier encoding
        # ----------------------------
        time_vec = torch.stack([hold, flight], dim=-1)  # (B, L, 3)
        time_feat = self.time_encoders(time_vec)  # (B, L, 2D)
        # ----------------------------
        # 2. Key embedding
        # ----------------------------
        key_feat = self.key_embedding(keys)  # (B, L, K)
        # ----------------------------
        # 3. Combine and project
        # ----------------------------
        encoded_x = self.input_proj(torch.cat([time_feat, key_feat], dim=-1))
        encoded_x = encoded_x + self.pos_enc[:, :encoded_x.size(1), :]

        # 4. Transformer
        attn_mask = (mask == 0)
        embedding = self.transformer(encoded_x, src_key_padding_mask=attn_mask)

        # 5. Masked mean pooling
        valid = mask.unsqueeze(-1)
        embedding = (embedding * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-8)

        # 6. Optional projection + normalize
        if self.use_projector:
            embedding = self.projector(embedding)

        return norm_embeddings(embedding)