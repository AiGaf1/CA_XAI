import pytorch_lightning as pl
import conf
import torch
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d  # For 1D Gaussian blur

from models.TTC import run_tcc_experiment, UserBaselineKNN
from main_train import create_data_module
from utils.visualization import visualize_keystrokes
from utils.tools import setup_logger
from models.Litmodel import KeystrokeLitModel
from models.transformer import Transformer_LTE
from utils.metrics import compute_distance

logger = setup_logger("main")
pl.seed_everything(conf.seed, workers=True, verbose=True)


def create_insertion(samples):
    """Create modified sample with illegitimate insertion."""
    L_legit_1 = samples['legit_mask_1'].sum().item()
    L_illegit = samples['illegit_mask'].sum().item()

    # Determine insertion parameters
    insert_len = random.randint(15, min(30, L_legit_1, L_illegit))
    insert_start = random.randint(0, L_legit_1 - insert_len)

    # Create modified sample
    mod_x = samples['legit_x_1'].clone()
    mod_x[insert_start:insert_start + insert_len] = samples['illegit_x'][:insert_len]

    print(f"Inserted at position {insert_start}, length {insert_len}")

    return {
        'x': mod_x,
        'mask': samples['legit_mask_1'].clone(),
        'start': insert_start,
        'length': insert_len
    }

def prepare_samples(batch, device, ):
    """Extract and prepare samples from batch."""
    (x1, mask1), (x2, mask2), labels, (u1, u2) = batch

    # Move to device
    x1, mask1 = x1.to(device), mask1.to(device)
    x2, mask2 = x2.to(device), mask2.to(device)

    # Find legitimate (same user) samples
    same_user_idx = [i for i, label in enumerate(labels) if label == 0]
    legit_id1 = same_user_idx[0]
    legit_id2 = same_user_idx[1]

    # Find illegitimate (different user) sample
    diff_user_idx = [i for i, label in enumerate(labels) if label == 1]
    illegit_id = diff_user_idx[0]

    return {
        'legit_x_1': x1[legit_id1].cpu(),
        'legit_mask_1': mask1[legit_id1].cpu(),
        'legit_x_2': x1[legit_id2].cpu(),
        'legit_mask_2': mask1[legit_id2].cpu(),
        'illegit_x': x2[illegit_id].cpu(),
        'illegit_mask': mask2[illegit_id].cpu(),
        'user_ids': (u1[legit_id1], u1[legit_id2])
    }


def compute_sample_distances(model, samples, insertion):
    """Compute embedding distances between samples."""
    device = next(model.parameters()).device

    # Get embeddings
    emb_legit_1 = model(
        samples['legit_x_1'].unsqueeze(0).to(device),
        samples['legit_mask_1'].unsqueeze(0).to(device)
    )
    emb_illegit = model(
        samples['illegit_x'].unsqueeze(0).to(device),
        samples['illegit_mask'].unsqueeze(0).to(device)
    )

    emb_legit_2 = model(
        samples['legit_x_2'].unsqueeze(0).to(device),
        samples['legit_mask_2'].unsqueeze(0).to(device)
    )

    emb_mod = model(
        insertion['x'].unsqueeze(0).to(device),
        insertion['mask'].unsqueeze(0).to(device)
    )

    # Compute distances
    distance_U1vsU1_diff_samples = compute_distance(emb_legit_1, emb_legit_2, 'cosine')
    distance_U1vsU1_mod_sample = compute_distance(emb_legit_2, emb_mod, 'cosine')
    distance_U1vsU2 = compute_distance(emb_legit_1, emb_illegit, 'cosine')

    print(f'Distance between same users: user1_sample1 VS user1_sample2: {distance_U1vsU1_diff_samples}')
    print(f'Distance with modified sample: user1_sample1 VS (user1_sample1 + user2_sample1): {distance_U1vsU1_mod_sample}')
    print(f'Distance between different users: user1_sample1 VS user2_sample1: {distance_U1vsU2}')
    return {
        'U1vsU1': distance_U1vsU1_diff_samples,
        'U1vsU1U2': distance_U1vsU1_mod_sample,
        'U1vsU2' : distance_U1vsU2
    }

def one_d_occlusion(x, mask, p, s):
    """
    Adapted Algorithm 1: Systematic 1D Occlusion.
    Input: x [L, 3] (hold, flight, keys), mask [L], p (patch size), s (stride)
    Output: list of occluded x, list of masks M (1D, 1 where occluded)
    """
    L = x.shape[0]
    occluded_xs = []
    occ_masks = []

    loc = 0
    while loc + p <= L:
        occ_x = x.clone()
        occ_x[loc:loc + p] = 0  # Occlude to 0
        occluded_xs.append(occ_x)

        M = torch.zeros(L, dtype=torch.float32)
        M[loc:loc + p] = 1
        occ_masks.append(M)

        loc += s

    return occluded_xs, occ_masks


def generate_similarity_map(model, device, orig_x1, mask1, orig_x2, mask2, patch_sizes=[7, 14, 28], s=5):
    """
    Adapted Method 3: Generate 1D similarity map for pair (x1, x2)
    """
    emb1 = model(orig_x1.unsqueeze(0).to(device), mask1.unsqueeze(0).to(device))[0]
    emb2 = model(orig_x2.unsqueeze(0).to(device), mask2.unsqueeze(0).to(device))[0]
    d_orig = 1 - F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()  # Distance

    similarity_maps = []

    for p in patch_sizes:
        occ_xs1, occ_ms1 = one_d_occlusion(orig_x1, mask1, p, s)
        occ_xs2, _ = one_d_occlusion(orig_x2, mask2, p, s)

        N = len(occ_xs1)
        if N == 0:
            continue

        ds = []
        for i in range(N):
            occ_emb1 = model(occ_xs1[i].unsqueeze(0).to(device), mask1.unsqueeze(0).to(device))[0]
            occ_emb2 = model(occ_xs2[i].unsqueeze(0).to(device), mask2.unsqueeze(0).to(device))[0]
            d_i = 1 - F.cosine_similarity(occ_emb1.unsqueeze(0), occ_emb2.unsqueeze(0)).item()
            ds.append(d_i)

        L = orig_x1.shape[0]
        S_p = np.zeros(L)
        count = np.zeros(L)
        for i in range(N):
            deviation = ds[i] - d_orig
            M_i = occ_ms1[i].cpu().numpy()
            S_p += deviation * M_i
            count += M_i

        S_p = np.divide(S_p, count, where=count != 0)
        similarity_maps.append(S_p)

    if not similarity_maps:
        raise ValueError("No valid occlusions")

    weights = [1 / (p ** 2 * len(patch_sizes)) for p in patch_sizes]
    S_bar = np.sum([S * w for S, w in zip(similarity_maps, weights)], axis=0)

    S_bar = gaussian_filter1d(S_bar, sigma=s)

    if S_bar.max() != S_bar.min():
        S_bar = 2 * (S_bar - S_bar.min()) / (S_bar.max() - S_bar.min()) - 1

    return S_bar, d_orig


def visualize_1d_map(S_bar, insert_start, insert_len):
    fig, ax = plt.subplots(figsize=(10, 2))
    positions = np.arange(len(S_bar))
    colors = ['red' if v < 0 else 'green' for v in S_bar]
    ax.bar(positions, S_bar, color=colors, alpha=0.7)
    ax.axvspan(insert_start, insert_start + insert_len, color='blue', alpha=0.3, label='Inserted Segment')
    ax.set_title('1D Similarity Map (Red: Dissimilar, Green: Similar)')
    ax.set_xlabel('Position')
    ax.set_ylabel('Deviation')
    ax.legend()
    plt.savefig('similarity_map.png')
    plt.close()
    print("Map saved as 'similarity_map.png'")


def visualize_results(samples, insertion, model, device):
    """Generate all visualizations."""
    # Visualize original and modified keystrokes
    visualize_keystrokes(
        samples['legit_x_2'].to(device),
        samples['legit_mask_2'].to(device),
        samples['user_ids'][1]
    )
    visualize_keystrokes(
        insertion['x'].to(device),
        insertion['mask'].to(device),
        samples['user_ids'][0]
    )

    # Generate similarity map
    S_bar, _ = generate_similarity_map(
        model, device,
        samples['legit_x_2'].to(device),
        samples['legit_mask_2'].to(device),
        samples['illegit_x'].to(device),
        samples['illegit_mask'].to(device)
    )
    visualize_1d_map(S_bar, insertion['start'], insertion['length'])

def build_windows(session_tensor, window_size):
    windows = []
    for start in range(0, len(session_tensor) - window_size + 1):
        w = session_tensor[start:start + window_size]
        windows.append(w)
    return torch.stack(windows)


def get_user_sessions_sample(dataloader):
    """Build map of all users from entire dataloader (x1 AND x2)."""

    user_sessions_map = {}

    for batch in dataloader:
        (x1, mask1), (x2, mask2), labels, (u1, u2), (_, _) = batch

        batch_size = len(u1)

        for i in range(batch_size):

            uid1 = u1[i]  # No .item() needed
            if uid1 not in user_sessions_map:
                user_sessions_map[uid1] = []

            user_sessions_map[uid1].append({
                'x': x1[i],
                'mask': mask1[i]
            })

            uid2 = u2[i]  # No .item() needed
            if uid2 not in user_sessions_map:
                user_sessions_map[uid2] = []

            user_sessions_map[uid2].append({
                'x': x2[i],
                'mask': mask2[i]
            })

    return user_sessions_map


def get_user_data(user_sessions_map, user_id, exclude_session=None):
    """
    Get all sessions and masks for a specific user, excluding a specific session tensor.
    """
    if isinstance(user_id, torch.Tensor):
        user_id = user_id.item()

    if user_id not in user_sessions_map:
        print(f"User {user_id} not found in dataset")
        return None, None

    sessions_data = user_sessions_map[user_id]
    # print(f"All Sessions of this user {user_id}: {sessions_data}")
    all_sessions = []
    all_masks = []

    for session in sessions_data:
        # Skip if this session matches the one to exclude
        if exclude_session is not None and torch.equal(session['x'], exclude_session):
            print("Session found and excluded")
            continue
        all_sessions.append(session['x'])
        all_masks.append(session['mask'])

    if not all_sessions:
        print(f"No sessions remaining for user {user_id} after exclusion")
        return None, None

    all_sessions_tensor = torch.stack(all_sessions)
    all_masks_tensor = torch.stack(all_masks)

    return all_sessions_tensor[:, :conf.window_size, :], all_masks_tensor[:, :conf.window_size]

def run_experiment(file_path: str):
    # =====================
    # Configuration
    # =====================
    sequence_length = 64
    window_size = 16 # length of session  for NN input
    max_samples = 6
    min_sessions_per_user = 4
    ckpt_path = "Keystroke-XAI/20260130_1234/checkpoints/mobile-651-1.50.ckpt"
    # ckpt_path = "Keystroke-XAI/20260211_0013/checkpoints/mobile-961-3.77.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =====================
    # Data
    # =====================
    dm = create_data_module(
        file_path,
        min_session_length=sequence_length,
        sequence_length=sequence_length,
        min_sessions_per_user=min_sessions_per_user

    )
    dm.setup(None)

    # Build user-session map once
    user_sessions_map = get_user_sessions_sample(dm.val_dataloader())

    # =====================
    # Model
    # =====================
    base_model = Transformer_LTE(
        periods_dict=dm.min_max,
        use_projector=conf.use_projector,
        window_size=conf.window_size,
    )

    model = KeystrokeLitModel.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        model=base_model,
    ).to(device)

    model.eval()

    # =====================
    # Batch inspection
    # =====================
    # loader = dm.val_same_sequence_dataloader()
    loader = dm.val_dataloader()
    (x1, mask1), (x2, mask2), labels, (u1, u2),  (_, _) = next(iter(loader))
    print(labels[0])
    x1, mask1 = x1.to(device), mask1.to(device)
    x2, mask2 = x2.to(device), mask2.to(device)

    # =====================
    # Experiment loop
    # =====================
    for sample_idx in range(max_samples):
        user_id = u1[sample_idx]  # Convert tensor to int
        # Debug: Check if user exists in map
        if user_id not in user_sessions_map:
            print(f"\n[Sample {sample_idx}] User {user_id} NOT found in user_sessions_map")
            print(f"Available user IDs: {sorted(user_sessions_map.keys())[:20]}")
            print(f"Total users in map: {len(user_sessions_map)}")
            continue

        session = torch.cat([x1[sample_idx], x2[sample_idx]], dim=0)
        tau_star_event = x1[sample_idx].shape[0]

        visualize_keystrokes(
            session,
            torch.ones(session.shape[0], dtype=torch.bool),
            user="Different Users" if labels[sample_idx] else "Same User",
            vline_index=tau_star_event,
        )

        if tau_star_event < window_size:
            print(f"[Sample {sample_idx}] Skipping: tau_star_event ({tau_star_event}) < window_size ({window_size})")
            continue

        # Build sliding windows
        windows = build_windows(session, window_size=window_size)
        window_masks = torch.ones(
            windows.shape[0], window_size, dtype=torch.bool, device=device
        )

        # Embeddings (attack session)
        embeddings = model(windows, window_masks).detach().cpu().numpy()

        # Get baseline user embeddings
        user_sessions, user_masks = get_user_data(
            user_sessions_map,
            user_id,  # Already converted to int
            x1[sample_idx].cpu(),
        )

        # Handle None case
        if user_sessions is None or user_masks is None:
            print(f"[Sample {sample_idx}] No valid sessions for user {user_id} after exclusion")
            print(f"Sessions in map for this user: {len(user_sessions_map[user_id])}")
            continue

        # windows = build_windows(session, window_size=window_size)
        # window_masks = torch.ones(
        #     windows.shape[0], window_size, dtype=torch.bool, device=device
        # )
        legitimate_embeddings = model(
            user_sessions.to(device),
            user_masks.to(device),
        ).detach().cpu()

        baseline = UserBaselineKNN(legitimate_embeddings)
        tau_star_window = max(0, tau_star_event - window_size + 1)

        run_tcc_experiment(
            baseline,
            embeddings,
            tau_star_window,
            label=labels[sample_idx],
            window_size=window_size
        )

    #
    # # Compute distances
    # distances = compute_sample_distances(loaded_model, samples, mod_sample)
    #
    # visualize_results(samples, mod_sample, loaded_model, device)

if __name__ == "__main__":
    file_path = f'data/{conf.scenario}/{conf.scenario}_dev_set.npy'
    run_experiment(file_path)    # wandb.finish()