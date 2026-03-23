import torch
import torch.nn as nn
from models.fourier_encoding import LearnableFourierFeatures
from models.cnn import norm_embeddings
class KeystrokeTransformer(nn.Module):
    def __init__(self, periods_dict, output_size=64, hidden_size=128,
                 window_size=50, vocab_size=256, key_emb_dim=16, use_projector=False,
                 num_layers=4, num_heads=2, ff_dim=512, dropout=0.2, n_periods=16,
                 use_pos_enc=True, use_mste=True):
        super().__init__()
        self.use_projector = use_projector
        self.use_pos_enc = use_pos_enc
        self.use_mste = use_mste
        self.d_model = hidden_size

        self.key_embedding = nn.Embedding(vocab_size, key_emb_dim)
        if use_mste:
            self.time_encoders = LearnableFourierFeatures(periods_dict, num_features=n_periods)
            input_size = self.time_encoders.d_out + key_emb_dim
        else:
            input_size = 2 + key_emb_dim  # raw hold + flight

        self.input_proj = nn.Linear(input_size, self.d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, window_size, self.d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        if use_projector:
            self.projector = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(self.d_model, output_size, bias=False)
            )
        self.embedding_dim = output_size if use_projector else self.d_model

    def forward(self, x, mask):
        x, mask = x.float(), mask.float()
        hold, flight, keys = x.unbind(dim=-1)
        keys = keys.long()

        # 1. Time encoding
        time_vec = torch.stack([hold, flight], dim=-1)  # (B, L, 2)
        time_feat = self.time_encoders(time_vec) if self.use_mste else time_vec  # (B, L, 2D) or (B, L, 2)
        # ----------------------------
        # 2. Key embedding
        # ----------------------------
        key_feat = self.key_embedding(keys)  # (B, L, K)
        # ----------------------------
        # 3. Combine and project
        # ----------------------------
        encoded_x = self.input_proj(torch.cat([time_feat, key_feat], dim=-1))
        if self.use_pos_enc:
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

    def get_embedding_dim(self) -> int:
        return self.embedding_dim