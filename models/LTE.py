import torch.nn as nn
import torch

class LTEOrig(nn.Module):
    def __init__(self, init_periods=[], proj_dim=None, eps=1e-5):
        super().__init__()

        init_periods = init_periods.detach().clone().to(torch.float32)
        # init_periods = init_periods.clamp(min=eps)
        # self.init_freqs = nn.Parameter(torch.log(init_periods))
        self.register_buffer("init_freqs", init_periods)

        self.scales = nn.Parameter(torch.ones_like(init_periods))
        # self.phases = nn.Parameter(torch.zeros_like(init_periods))

        self.d_out = 2 * len(init_periods)
        if proj_dim is not None:
            self.proj = nn.Linear(self.d_out, proj_dim)
        else:
            self.proj = nn.Identity()
        self.eps = eps

    def forward(self, t):
        """
        t: (B, S) time values in seconds
        returns: (B, S, proj_dim or 2*n_freqs)
        """
        # periods = torch.exp(self.init_freqs.clamp(min=self.eps))
        periods = self.init_freqs
        freqs = 2 * torch.pi / periods  # (n_freqs,)
        # angles = t[..., None] * freqs  + self.phases        # (B, S, n_freqs)
        angles = t[..., None] * freqs

        sin_part = angles.sin() * self.scales
        cos_part = angles.cos() * self.scales

        enc = torch.cat([sin_part, cos_part], dim=-1)
        return self.proj(enc)
