import torch.nn as nn
import torch

import math

class LearnableFourierFeatures(nn.Module):
    def __init__(self, feature_dict: dict, num_features: int):
        """
        feature_dict: dict where key=feature_name, value={"min": float, "max": float}
        num_features: number of Fourier frequencies per feature
        """
        super().__init__()

        self.feature_names = list(feature_dict.keys())
        self.num_features = num_features
        self.input_dim = len(self.feature_names)

        # Initialize a frequency matrix (M x D)
        period_list = []
        for feat in self.feature_names:
            min_f, max_f = feature_dict[feat]["min"], feature_dict[feat]["max"]
            period = torch.logspace(math.log10(min_f), math.log10(max_f)/2, steps=num_features)
            period_list.append(period)
        period_init = torch.stack(period_list, dim=0)  # (M, D)

        freq = 2 * torch.pi / period_init
        self.register_buffer("freq", freq)

        init = torch.randn(self.input_dim, self.num_features) * 0.1
        self.scales_raw = nn.Parameter(init)

        # Amplitude gate: sigmoid(a) ∈ (0,1), init=0 → sigmoid(0)=0.5
        # L1 on amplitude pushes frequencies toward 0 → sparsity
        # self.amplitude_raw = nn.Parameter(torch.zeros(self.input_dim, self.num_features))
        self.d_out = 2 * self.input_dim * num_features

    def get_scales(self) -> torch.Tensor:
        return torch.sigmoid(self.scales_raw)

    # def get_amplitude(self) -> torch.Tensor:
    #     """Return amplitude gate values: sigmoid(a) ∈ (0,1). Use for L1 sparsity."""
    #     return torch.sigmoid(self.amplitude_raw)

    def forward(self, x):
        """
        x: (B, L, M)
        return: (B, L, M*2*D)
        """

        x = x.unsqueeze(-1)                     # (B, L, M, 1)
        scales = self.get_scales()
        proj = x * self.freq * scales          # (B, L, M, D)
        # amplitude = self.get_amplitude()       # (M, D)
        fourier = torch.stack(
            [proj.sin(), proj.cos()],
            dim=-1                              # (B, L, M, D, 2)
        )      

        return fourier.flatten(start_dim=-3)