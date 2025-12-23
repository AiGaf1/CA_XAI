import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.fft as fft

class LTEOrig(nn.Module):
    def __init__(self, init_periods):
        super().__init__()

        init_periods = init_periods.detach().clone().to(torch.float32)
        init_freqs = 2 * torch.pi / init_periods
        self.register_buffer("freqs", init_freqs)

        self.scales_raw = nn.Parameter(torch.ones_like(init_periods))
        # self.phases = nn.Parameter(torch.zeros_like(init_periods))
        self.d_out = 2 * len(init_periods)

    def forward(self, t):
        """
        t: (Batch, Sequence Length) time values in seconds of each feature
        returns: (B, S, 2*n_freqs)
        """
        # angles = t[..., None] * freqs  + self.phases        # (B, S, n_freqs)
        angles = t[..., None] * self.freqs

        sin_part = angles.sin()
        cos_part = angles.cos()
        enc = torch.cat([sin_part, cos_part], dim=-1)
        sigmoid_module = nn.Sigmoid()
        scales = sigmoid_module(self.scales_raw)
        scales = scales.repeat(1, 2) #we repeat scales 2 times for sin and cos
        return enc * scales

class MultiplyFreqs(nn.Module):
    def __init__(self, freqs):
        super().__init__()
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        return t[..., None] * self.freqs

class SinCos(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, angles):
        sin_part = angles.sin()
        cos_part = angles.cos()
        return torch.cat([sin_part, cos_part], dim=-1)

class ScaledSigmoid(nn.Module):
    def __init__(self, num_freqs):
        super().__init__()
        self.scales_raw = nn.Parameter(torch.ones(num_freqs))
        self.sigmoid_module = nn.Sigmoid()

    def forward(self, enc):
        scales = self.sigmoid_module(self.scales_raw)
        scales = scales.repeat(1, 2)
        return enc * scales

# To create the equivalent nn.Sequential, assuming init_periods is provided
def create_lte_sequential(init_periods):
    init_periods = init_periods.detach().clone().to(torch.float32)
    init_freqs = 2 * torch.pi / init_periods
    num_freqs = len(init_periods)

    model = nn.Sequential(
        MultiplyFreqs(init_freqs),
        SinCos(),
        ScaledSigmoid(num_freqs)
    )
    model.d_out = 2 * num_freqs  # Attach d_out as an attribute for compatibility
    return model


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
