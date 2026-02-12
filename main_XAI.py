import numpy as np
import wandb
import torch
import random

from main_train import create_trainer, run_predictions, compute_distance, create_data_module
from data.dataset import KeystrokeDataModule
from utils.tools import setup_logger
from utils.visualization import visualize_keystrokes, compare_two_users
from models.Litmodel import KeystrokeLitModel
from models.transformer import Transformer_LTE
import pytorch_lightning as pl
import conf
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d  # For 1D Gaussian blur
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

logger = setup_logger("main")
pl.seed_everything(conf.seed, workers=True, verbose=True)


def create_hybrid_vectors(
        legitimate_vec: torch.Tensor,
        illegitimate_vec: torch.Tensor,
        patch_size: int,
        stride: int = 1
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Create hybrid vectors by replacing segments of legitimate vector with illegitimate segments.

    Args:
        legitimate_vec: Original legitimate user vector (length 128)
        illegitimate_vec: Illegitimate user vector to inject parts from
        patch_size: Size of the segment to replace
        stride: Step size for sliding window

    Returns:
        hybrid_vectors: List of hybrid vectors with different segments replaced
        masks: List of binary masks indicating which positions were replaced
    """
    vec_length = legitimate_vec.shape[0]
    hybrid_vectors = []
    masks = []

    # Slide window across vector
    for start_idx in range(0, vec_length - patch_size + 1, stride):
        end_idx = start_idx + patch_size

        # Create hybrid vector
        hybrid = legitimate_vec.clone()
        hybrid[start_idx:end_idx] = illegitimate_vec[start_idx:end_idx]

        # Create mask (1 where replaced, 0 elsewhere)
        mask = torch.zeros(vec_length, device=legitimate_vec.device)
        mask[start_idx:end_idx] = 1.0

        hybrid_vectors.append(hybrid)
        masks.append(mask)

    return hybrid_vectors, masks


def generate_explanation_map_method3(
        model,
        legitimate_input: Tuple[torch.Tensor, torch.Tensor],  # (x, mask)
        illegitimate_input: Tuple[torch.Tensor, torch.Tensor],  # (x, mask)
        patch_sizes: List[int] = [7, 14, 28],
        stride: int = 5,
        distance_metric: str = 'cosine'
) -> Dict:
    """
    Generate explanation map using Method 3 from the paper (co-located occlusions).

    This identifies which segments of the legitimate vector, when replaced with
    illegitimate segments, contribute most to dissimilarity.

    Args:
        model: The trained model
        legitimate_input: (x, mask) tuple for legitimate user
        illegitimate_input: (x, mask) tuple for illegitimate user
        patch_sizes: List of segment sizes to test
        stride: Step size for sliding window
        distance_metric: 'cosine' or 'euclidean'

    Returns:
        Dictionary containing explanation map and metadata
    """
    device = legitimate_input[0].device
    x_legit, mask_legit = legitimate_input
    x_illegit, mask_illegit = illegitimate_input

    # Get original embeddings
    with torch.no_grad():
        emb_legit = model(x_legit.unsqueeze(0), mask_legit.unsqueeze(0)).squeeze(0)
        emb_illegit = model(x_illegit.unsqueeze(0), mask_illegit.unsqueeze(0)).squeeze(0)

    embedding_dim = emb_legit.shape[0]

    # Compute original distance
    distance_original = compute_distance(
        emb_legit.unsqueeze(0),
        emb_illegit.unsqueeze(0),
        distance_metric
    )

    print(f"Original distance (legitimate vs illegitimate): {distance_original:.4f}")

    # Storage for all deviation maps
    all_deviation_maps = []
    all_weights = []

    # Process each patch size
    for patch_size in patch_sizes:
        print(f"\nProcessing patch size: {patch_size}")

        # Create hybrid embeddings by replacing segments
        hybrid_embeddings, masks = create_hybrid_vectors(
            emb_legit, emb_illegit, patch_size, stride
        )

        # Compute distances for each hybrid
        deviations = []
        for hybrid_emb in hybrid_embeddings:
            dist = compute_distance(
                hybrid_emb.unsqueeze(0),
                emb_illegit.unsqueeze(0),
                distance_metric
            )

            # Deviation = how much the distance changed
            # Positive deviation = vectors became MORE similar (dissimilar segment found)
            # Negative deviation = vectors became LESS similar (similar segment)
            deviation = distance_original - dist
            deviations.append(deviation)

        # Create deviation map by aggregating weighted masks
        deviation_map = torch.zeros(embedding_dim, device=device)
        for deviation, mask in zip(deviations, masks):
            deviation_map += deviation * mask

        # Normalize by number of times each position was covered
        coverage_count = torch.zeros(embedding_dim, device=device)
        for mask in masks:
            coverage_count += mask
        coverage_count = torch.clamp(coverage_count, min=1)  # Avoid division by zero
        deviation_map = deviation_map / coverage_count

        all_deviation_maps.append(deviation_map)
        all_weights.append(patch_size ** 2)  # Weight by area (as in paper)

        print(f"Generated {len(hybrid_embeddings)} hybrid vectors")
        print(f"Deviation range: [{min(deviations):.4f}, {max(deviations):.4f}]")

    # Combine multi-scale deviation maps (weighted average)
    total_weight = sum(all_weights)
    combined_deviation_map = sum(
        dev_map * weight / total_weight
        for dev_map, weight in zip(all_deviation_maps, all_weights)
    )

    # Normalize to [-1, 1] for visualization
    max_abs = torch.abs(combined_deviation_map).max()
    if max_abs > 0:
        normalized_map = combined_deviation_map / max_abs
    else:
        normalized_map = combined_deviation_map

    return {
        'deviation_map': combined_deviation_map.cpu().numpy(),
        'normalized_map': normalized_map.cpu().numpy(),
        'distance_original': distance_original,
        'patch_sizes': patch_sizes,
        'stride': stride,
        'embedding_dim': embedding_dim
    }


def visualize_explanation_map(
        explanation_result: Dict,
        title: str = "Dissimilarity Explanation Map",
        save_path: str = None
):
    """
    Visualize the explanation map showing which segments contribute to dissimilarity.

    Args:
        explanation_result: Output from generate_explanation_map_method3
        title: Plot title
        save_path: Optional path to save figure
    """
    normalized_map = explanation_result['normalized_map']
    embedding_dim = explanation_result['embedding_dim']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Heatmap
    im = ax1.imshow(
        normalized_map.reshape(1, -1),
        cmap='RdYlGn',  # Red (dissimilar) to Green (similar)
        aspect='auto',
        vmin=-1,
        vmax=1
    )
    ax1.set_title(f'{title}\n(Red = Dissimilar segments, Green = Similar segments)',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Embedding Dimension', fontsize=10)
    ax1.set_yticks([])
    ax1.set_xlim(-0.5, embedding_dim - 0.5)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, orientation='vertical', pad=0.02)
    cbar.set_label('Deviation Score', rotation=270, labelpad=20)

    # Plot 2: Line plot
    x_indices = np.arange(embedding_dim)
    ax2.plot(x_indices, normalized_map, linewidth=2, color='navy', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.fill_between(x_indices, 0, normalized_map,
                     where=(normalized_map > 0),
                     color='red', alpha=0.3, label='Dissimilar')
    ax2.fill_between(x_indices, 0, normalized_map,
                     where=(normalized_map <= 0),
                     color='green', alpha=0.3, label='Similar')

    ax2.set_title('Deviation Profile', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Embedding Dimension', fontsize=10)
    ax2.set_ylabel('Normalized Deviation', fontsize=10)
    ax2.set_xlim(-0.5, embedding_dim - 0.5)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    # Add metadata
    info_text = (
        f"Original Distance: {explanation_result['distance_original']:.4f}\n"
        f"Patch Sizes: {explanation_result['patch_sizes']}\n"
        f"Stride: {explanation_result['stride']}"
    )
    ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()

    # Print top dissimilar segments
    top_k = 5
    top_indices = np.argsort(normalized_map)[-top_k:][::-1]
    print(f"\nTop {top_k} most dissimilar embedding dimensions:")
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. Dimension {idx}: {normalized_map[idx]:.4f}")

def run_experiment(file_path: str):
    FULL_NAME = f'{conf.epochs}_{conf.scenario}'
    # init_launch()
    dm = create_data_module(file_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dm.setup(None)

    small_loader = dm.val_dataloader()

    # print((x1[0], x2[0]), labels[0], (u1[0], u2[0]))
    ckpt_path = "Keystroke-XAI/20260130_1234/checkpoints/mobile-651-1.50.ckpt"

    nn_model = Transformer_LTE(periods_dict=dm.init_periods, use_projector=conf.use_projector)
    loaded_model = KeystrokeLitModel.load_from_checkpoint(
        checkpoint_path = ckpt_path,
        model=nn_model
    )
    loaded_model.eval()
    loaded_model.to(device)


    (x1, mask1), (x2, mask2), labels, (u1, u2) = next(iter(small_loader))
    x1 = x1.to(device)
    x2 = x2.to(device)
    mask1 = mask1.to(device)
    mask2 = mask2.to(device)

    id1  = 3  # session_id
    id2  = 3  # session_id

    print(f"Analyzing User {u1[id1].item()} (legitimate) vs User {u2[id1].item()} (illegitimate)")
    print(f"True label (should be 1 for imposter): {labels[id2].item()}")

    embeddings1 = loaded_model(x1, mask1)  # This calls forward()
    embeddings2 = loaded_model(x2, mask2)
    distance = compute_distance(
        embeddings1[id1].unsqueeze(0),
        embeddings2[id2].unsqueeze(0),
        'cosine'
    )
    print(f"\nOriginal distance: {distance:.4f}")
    compare_two_users(x1[id1], mask1[id1], u1[id1], x2[id2], mask2[id2], u2[id2], distance)

    # Generate explanation map using Method 3
    print("\n" + "=" * 60)
    print("GENERATING EXPLANATION MAP (Method 3)")
    print("=" * 60)

    explanation_result = generate_explanation_map_method3(
        model=loaded_model,
        legitimate_input=(x1[id1], mask1[id1]),
        illegitimate_input=(x2[id2], mask2[id2])
    )

    # Visualize results
    visualize_explanation_map(
        explanation_result,
        title=f"User {u1[id1].item()} vs User {u2[id1].item()}",
        save_path=f"explanation_map_user{u1[id1].item()}_vs_user{u2[id1].item()}.png"
    )

    return explanation_result

    # compare_two_users(x1[id1], mask1[id1], u1[id1], x2[id2], mask2[id2], u2[id2], distance)
    # print(distance)
    # print(labels[id1])

if __name__ == "__main__":
    file_path = f'data/{conf.scenario}/{conf.scenario}_dev_set.npy'
    run_experiment(file_path)
    # wandb.finish()