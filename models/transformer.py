import torch
import torch.nn as nn
import conf
from models.LTE import LTEOrig, create_lte_sequential
import torch.nn.functional as F
from collections import OrderedDict

def norm_embeddings(embeddings):
    return F.normalize(embeddings, dim=-1, p=2)

class Transformer_LTE(nn.Module):
    def __init__(self, periods_dict, output_size=512, hidden_size=128,
                 sequence_length=128, vocab_size=256, key_emb_dim=16, use_projector=False,
                 num_layers=4, num_heads=2, ff_dim=512, dropout=0.1):
        super().__init__()
        self.time_encoders = nn.ModuleDict()
        self.key_embedding = nn.Embedding(vocab_size, key_emb_dim)
        self.use_projector = use_projector
        self.sequence_length = sequence_length
        self.d_model = hidden_size
        for feat, periods in periods_dict.items():
            self.time_encoders[feat] = LTEOrig(init_periods=periods)

        input_size = sum(encoder.d_out for encoder in self.time_encoders.values()) + key_emb_dim

        self.input_proj = nn.Linear(input_size, self.d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, sequence_length, self.d_model))

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
                ("bn", nn.BatchNorm1d(self.d_model)),
                ("drop", nn.Dropout(p=0.2)),
                ("fc_out", nn.Linear(self.d_model, output_size, bias=False))
            ]))

        self._init_weights()

    def _init_weights(self):
        """Proper weight initialization for better gradient flow"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.25)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, mask):
        x = x.float()
        mask = mask.float()

        hold, flight, keys = x.unbind(dim=-1)
        keys = keys.long()
        encoded_x = [
            self.time_encoders["hold"](hold),
            self.time_encoders["flight"](flight),
            self.key_embedding(keys)  # key
        ]
        encoded_x = torch.cat(encoded_x, dim=-1)  # (B, L, input_size)

        encoded_x = self.input_proj(encoded_x)  # (B, L, d_model)
        encoded_x = encoded_x + self.pos_enc[:, :encoded_x.size(1), :]

        key_padding_mask = (mask == 0)  # (B, L), True for positions to ignore

        features = self.transformer(encoded_x, src_key_padding_mask=key_padding_mask)  # (B, L, d_model)

        # Masked mean pooling
        valid = mask.unsqueeze(-1)  # (B, L, 1)
        denom = valid.sum(dim=1, keepdim=True) + 1e-8  # (B, 1, 1)
        embedding = (features * valid).sum(dim=1) / denom.squeeze(-1)  # (B, d_model)

        if self.use_projector:
            embedding = self.projector(embedding)

        embedding = norm_embeddings(embedding)
        return embedding