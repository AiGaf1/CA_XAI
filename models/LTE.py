import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.fft as fft
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
            period = torch.logspace(math.log10(min_f), math.log10(max_f), steps=num_features)
            period_list.append(period)
        period_init = torch.stack(period_list, dim=0)  # (M, D)

        freq = 2 * torch.pi / period_init
        self.register_buffer("freq", freq)

        # learnable amplitude
        self.scales_raw = nn.Parameter(
            torch.zeros(self.input_dim, self.num_features)  # (M, D)
        )

        self.d_out = 2 * self.input_dim * num_features

    def forward(self, x):
        """
        x: (B, L, M)
        return: (B, L, M*2*D)
        """

        x = x.unsqueeze(-1)                     # (B, L, M, 1)
        scales = torch.sigmoid(self.scales_raw) # bounded (0,1)
        proj = x * self.freq * scales          # (B, L, M, D)
        fourier = torch.stack(
            [proj.sin(), proj.cos()],
            dim=-1                              # (B, L, M, D, 2)
        )

        return fourier.flatten(start_dim=-3)



class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, n_freqs):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_freqs = n_freqs

        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels))

    def forward(self, x):
        x_ft = torch.fft.rfft(x)

        out_ft = torch.zeros(x.shape[0], self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.n_freqs] = torch.einsum("bix, iox->box", x_ft[:, :, :self.n_freqs], self.weights)

        # x = torch.fft.irfft(out_ft, n=x.size(-1))
        return out_ft
