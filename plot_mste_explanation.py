import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from data.Aalto.preprocessing import extract_features_classic

# ── Load checkpoint — desktop N=8 (EER=0.74%) ────────────────────────────────
CKPT = 'outputs/Keystroke-XAI/20260314_071108/checkpoints/desktop-868-0.74.ckpt'
state = torch.load(CKPT, map_location='cpu')['state_dict']

# freq[i, j] = 2π / p_j  (angular base frequency, stored in model)
# scales[i, j] = σ(ŵ_j) ∈ (0, 1)   (learnable sigmoid gate)
# ω_j = freq[i,j] * scales[i,j]     (effective angular frequency, eq. 5 in paper)
# p*_j = 2π / ω_j                   (learned effective period)
# amplitude[i, j] = σ(amp_raw[i,j]) (learned amplitude gate, eq. 6)
freq       = state['model._orig_mod.time_encoders.freq'].numpy()          # (2, N)
scales_raw = state['model._orig_mod.time_encoders.scales_raw']
amplitude_raw = state['model._orig_mod.time_encoders.amplitude_raw']
scales     = torch.sigmoid(scales_raw).numpy()                            # (2, N)
amplitude  = torch.sigmoid(amplitude_raw).numpy()                         # (2, N)

eff_freq    = freq * scales           # ω_j  (2, N)
eff_periods = 2 * np.pi / eff_freq   # p*_j (2, N), in seconds

# ── Load real session — 50 keystrokes ────────────────────────────────────────
raw  = np.load('data/Aalto/raw/desktop/desktop_dev_set.npy', allow_pickle=True).item()
user = list(raw.keys())[0]
sess = raw[user][list(raw[user].keys())[0]]
feats = extract_features_classic(sess)
ht_raw = feats[:50, 0]

n = len(ht_raw)
t = np.arange(n)
peak_i = np.argmax(ht_raw)

# ── MSTE (vectorised) ─────────────────────────────────────────────────────────
# Implements eq. (6): MSTE(t_i) = [sin(ω_j · t_i), cos(ω_j · t_i)]_{j=1}^N
# weighted by learned amplitude σ(â_j)
def mste_exact(signal, omega_arr, amp_arr):
    """
    signal:    (T,)
    omega_arr: (N,)  effective angular frequencies ω_j
    amp_arr:   (N,)  learned amplitudes σ(â_j)
    Returns:   (2N, T) — [sin row, cos row] interleaved per frequency
    """
    proj = omega_arr[:, None] * signal[None, :]       # (N, T)
    sin_enc = np.sin(proj) * amp_arr[:, None]         # (N, T)
    cos_enc = np.cos(proj) * amp_arr[:, None]         # (N, T)
    return np.stack([sin_enc, cos_enc], axis=1).reshape(-1, len(signal))  # (2N, T)

# HT encoder: index 0
ht_omega   = eff_freq[0]       # (N,)  ω_j for HT
ht_amp     = amplitude[0]      # (N,)  σ(â_j) for HT
ht_per     = eff_periods[0]    # (N,)  p*_j for HT
ht_mste    = mste_exact(ht_raw, ht_omega, ht_amp)   # (2N, T)

def fmt_period(p):
    """Format effective period p* (seconds) for axis labels."""
    if p < 1.0:
        return f'{p * 1000:.0f} ms'
    return f'{p:.2f} s'

# Select 4 representative frequencies covering behavioral time scales
# Effective HT periods (ms): ~2, ~6, ~17, ~79, ~229, ~622, ~1707, ~5481
# Chosen to match paper annotations: motor noise, finger alternation,
# word boundary, sentence boundary
# Indices [0, 2, 5, 6] → ~2 ms, ~17 ms, ~622 ms, ~1.7 s
SHOW_IDX    = [0, 2, 5, 6]
show_colors = ['#d62728', '#ff7f0e', '#1f77b4', '#2ca02c']
annotations = ['motor noise', 'finger alternation', 'word boundary', 'sentence boundary']

# ── Typography (IEEE double-column = 7.16 in full width) ─────────────────────
plt.rcParams.update({
    'font.family':     'serif',
    'font.serif':      ['Times New Roman', 'DejaVu Serif'],
    'font.size':        7,
    'axes.titlesize':   7.5,
    'axes.labelsize':   7,
    'xtick.labelsize':  6.5,
    'ytick.labelsize':  6.5,
    'axes.linewidth':   0.5,
    'xtick.major.width':0.5,
    'ytick.major.width':0.5,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'figure.dpi':     300,
    'pdf.fonttype':    42,
    'ps.fonttype':     42,
})

CBAR_SIZE = '4%'
CBAR_PAD  = '2%'

def _add_dummy_cbar(ax):
    make_axes_locatable(ax).append_axes('right', size=CBAR_SIZE, pad=CBAR_PAD).set_visible(False)

def _tighten_gap(ax_top, ax_bot, frac):
    pt = ax_top.get_position()
    pb = ax_bot.get_position()
    gap = pt.y0 - (pb.y0 + pb.height)
    ax_bot.set_position([pb.x0, pb.y0 + gap * frac, pb.width, pb.height])

fig = plt.figure(figsize=(7.16, 5.5))
gs  = gridspec.GridSpec(3, 1, figure=fig,
                        height_ratios=[1.4, 1.7, 1.8], hspace=0.85)

# ─────────────────────────────────────────────────────────────────────────────
# (a) Raw hold time
# ─────────────────────────────────────────────────────────────────────────────
y_ms   = ht_raw * 1000
peak_v = y_ms[peak_i]

ax_a = fig.add_subplot(gs[0])
_add_dummy_cbar(ax_a)

ax_a.semilogy(t, y_ms, color='#333333', lw=0.8,
              marker='o', ms=1.8, markerfacecolor='white',
              markeredgewidth=0.55, zorder=3)
ax_a.axvline(peak_i, color='#d62728', lw=0.65, ls='--', alpha=0.45, zorder=0)
ax_a.annotate(f'{peak_v:.0f} ms',
              xy=(peak_i, peak_v), xytext=(peak_i - 8, peak_v * 0.5),
              fontsize=6, color='#d62728',
              arrowprops=dict(arrowstyle='->', color='#d62728',
                              lw=0.65, shrinkA=2, shrinkB=2))

ax_a.set_xlim(-0.5, n - 0.5)
ax_a.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:.0f}'))
ax_a.set_ylabel(r'$\mathrm{HT}_k$ (ms)')
ax_a.tick_params(axis='x', labelbottom=False)
ax_a.set_title(r'(a)  Inter-key hold time $\mathrm{HT}_k$ (log scale)', pad=3, loc='left')
ax_a.spines[['top', 'right']].set_visible(False)
ax_a.grid(axis='y', lw=0.2, ls='--', color='#cccccc', zorder=0)

# ─────────────────────────────────────────────────────────────────────────────
# (b) MSTE heatmap
# Panel title matches paper eq. (6):
#   MSTE(HT_k) = [sin(ω_j · HT_k), cos(ω_j · HT_k)]  with ω_j = σ(ŵ_j)·(2π/p_j)
# ─────────────────────────────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[1])
cax_b = make_axes_locatable(ax_b).append_axes('right', size=CBAR_SIZE, pad=CBAR_PAD)

ht_mste_show = np.vstack([ht_mste[2*idx:2*idx+2] for idx in SHOW_IDX])
N_show = len(SHOW_IDX)
vabs = np.abs(ht_mste_show).max()
im = ax_b.imshow(
    ht_mste_show, aspect='auto', cmap='RdBu_r', vmin=-vabs, vmax=vabs,
    extent=[-0.5, n - 0.5, -0.5, 2*N_show - 0.5], origin='lower'
)
for i in range(1, N_show):
    ax_b.axhline(2*i - 0.5, color='white', lw=0.5)

ax_b.set_yticks([2*i + 0.5 for i in range(N_show)])
ax_b.set_yticklabels([fmt_period(ht_per[i]) for i in SHOW_IDX], fontsize=6.0)
ax_b.set_ylabel(r'Learned period $p^*_j$', labelpad=3)
ax_b.set_xlim(-0.5, n - 0.5)
ax_b.set_title(
    r'(b)  MSTE: $[\sin(\omega_j\,\mathrm{HT}_k),\,\cos(\omega_j\,\mathrm{HT}_k)]$,'
    r'  $\omega_j = \sigma(\hat{w}_j)\cdot(2\pi/p_j)$',
    pad=3, loc='left')
ax_b.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
ax_b.tick_params(labelbottom=False)
ax_b.axvline(peak_i, color='#d62728', lw=0.8, ls='--', alpha=0.55, zorder=3)

cbar = fig.colorbar(im, cax=cax_b)
cbar.set_ticks([-vabs, 0, vabs])
cbar.set_ticklabels([f'{-vabs:.2f}', '0', f'{vabs:.2f}'])
cbar.ax.tick_params(labelsize=6)
cbar.set_label('Encoding value', fontsize=6, labelpad=2)

for k in range(N_show):
    ax_b.plot(-1.1, 2*k + 0.5, marker='|', ms=6, mew=2.0,
              color=show_colors[k], clip_on=False, zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# (c) Four representative sin/cos components
# Amplitude-weighted: MSTE row = σ(â_j) · sin/cos(ω_j · HT_k)
# ─────────────────────────────────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[2])
_add_dummy_cbar(ax_c)

offset_step = 2.0
y_lo, y_hi = -1.5, len(SHOW_IDX) * offset_step + 0.3

for k, idx in enumerate(SHOW_IDX):
    sin_v = ht_mste[2*idx]
    cos_v = ht_mste[2*idx + 1]
    base  = k * offset_step
    c     = show_colors[k]
    ax_c.plot(t, base + sin_v, color=c, lw=0.9, ls='-',  alpha=0.95)
    ax_c.plot(t, base + cos_v, color=c, lw=0.9, ls='--', alpha=0.65)
    ax_c.axhline(base, color='#dddddd', lw=0.35, zorder=0)

ax_c.axvline(peak_i, color='#d62728', lw=0.8, ls='--', alpha=0.55, zorder=3)

ax_c.legend(handles=[
    Line2D([0],[0], color='#333333', lw=0.9, ls='-',
           label=r'$\sigma(\hat{a}_j)\sin(\omega_j\,\mathrm{HT}_k)$'),
    Line2D([0],[0], color='#333333', lw=0.9, ls='--',
           label=r'$\sigma(\hat{a}_j)\cos(\omega_j\,\mathrm{HT}_k)$'),
    Line2D([0],[0], color='#d62728', lw=0.8, ls='--', label=r'peak $\mathrm{HT}_k$'),
], fontsize=6.5, frameon=False, loc='upper right',
   bbox_to_anchor=(1.0, 1.10), ncol=3, columnspacing=0.8, handlelength=1.5)

ax_c.set_xlim(-0.5, n - 0.5)
ax_c.set_ylim(y_lo, y_hi)
ax_c.set_yticks([k * offset_step for k in range(len(SHOW_IDX))])
ax_c.set_yticklabels([fmt_period(ht_per[i]) for i in SHOW_IDX], fontsize=6.5)
for k, lbl in enumerate(ax_c.get_yticklabels()):
    lbl.set_color(show_colors[k])

for k, idx in enumerate(SHOW_IDX):
    base = k * offset_step
    ax_c.text(1.01, (base - y_lo) / (y_hi - y_lo),
              annotations[k], fontsize=6, color=show_colors[k],
              ha='left', va='center', transform=ax_c.transAxes)

ax_c.set_xlabel('Keystroke index $k$')
ax_c.set_ylabel('Encoding amplitude', labelpad=3)
ax_c.set_title(
    r'(c)  Amplitude-weighted components $\sigma(\hat{a}_j)[\sin,\cos](\omega_j\,\mathrm{HT}_k)$ per learned period $p^*_j$',
    pad=3, loc='left')
ax_c.spines[['top', 'right']].set_visible(False)
ax_c.grid(axis='x', lw=0.2, ls=':', color='#cccccc', zorder=0)

# ─────────────────────────────────────────────────────────────────────────────
fig.align_ylabels([ax_a, ax_b, ax_c])
_tighten_gap(ax_a, ax_b, 0.55)
_tighten_gap(ax_b, ax_c, 0.75)

os.makedirs('outputs', exist_ok=True)
fig.savefig('outputs/mste_explanation.pdf', bbox_inches='tight')
fig.savefig('outputs/mste_explanation.png', bbox_inches='tight', dpi=300)
print("Saved outputs/mste_explanation.{pdf,png}")
plt.show()
