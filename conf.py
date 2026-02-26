"""
Ablation Study Configuration
Define different experimental configurations here
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    """Single experiment configuration"""
    name: str = "Keystroke-XAI"

    # Data parameters
    window_size: int = 50
    min_session_length: int = 5
    min_sessions_per_user: int = 2
    train_split: float = 0.80
    val_split: float = 0.20
    test_split: float = 0.10

    # Model architecture
    model_type: str = "transformer" #cnn
    encoding_type: str = "lte"  # "lte", "fte", "clip"
    hidden_size: int = 128
    output_size: int = 512
    use_projector: bool = False

    # Transformer specific
    num_layers: int = 6
    num_heads: int = 2
    ff_dim: int = 512
    dropout: float = 0.1

    # Temporal encoding
    n_periods: int = 16

    # Loss function
    loss_type: str = "supcon"  # "supcon", "arcface", "triplet"

    # Augmentation
    use_augmentation: bool = True

    # Training
    epochs: int = 1000
    lr: float = 1e-3
    t_0: int = 2500

    # Batch settings
    batches_per_epoch_train: int = 256
    batches_per_epoch_val: int = 16
    samples_per_batch_train: int = 512
    samples_per_batch_val: int = 512

    #Data path
    scenario: str = 'desktop'
    file_path: str = f'data/Aalto/{scenario}/{scenario}_dev_set.npy'
    predict_file_path: str = f'data/{scenario}/{scenario}_test_sessions.npy'

    # Other
    seed: int = 42


FREQUENCY_EXPERIMENTS = {
    # LTE with different N_PERIODS
    "lte_freq_4": ExperimentConfig(
        name="lte_freq_4",
        encoding_type="lte",
        n_periods=4,
        loss_type="supcon",
        use_augmentation=True
    ),

    "lte_freq_8": ExperimentConfig(
        name="lte_freq_8",
        encoding_type="lte",
        n_periods=8,
        loss_type="supcon",
        use_augmentation=True
    ),

    "lte_freq_16": ExperimentConfig(
        name="lte_freq_16",
        encoding_type="lte",
        n_periods=16,
        loss_type="supcon",
        use_augmentation=True
    ),

    "lte_freq_32": ExperimentConfig(
        name="lte_freq_32",
        encoding_type="lte",
        n_periods=32,
        loss_type="supcon",
        use_augmentation=True
    ),

    "lte_freq_64": ExperimentConfig(
        name="lte_freq_64",
        encoding_type="lte",
        n_periods=64,
        loss_type="supcon",
        use_augmentation=True
    ),

    "lte_freq_128": ExperimentConfig(
        name="lte_freq_128",
        encoding_type="lte",
        n_periods=128,
        loss_type="supcon",
        use_augmentation=True
    ),

    # FTE (Fixed) with same frequencies for fair comparison
    "fte_freq_4": ExperimentConfig(
        name="fte_freq_4",
        encoding_type="fte",
        n_periods=4,
        loss_type="supcon",
        use_augmentation=True
    ),

    "fte_freq_8": ExperimentConfig(
        name="fte_freq_8",
        encoding_type="fte",
        n_periods=8,
        loss_type="supcon",
        use_augmentation=True
    ),

    "fte_freq_16": ExperimentConfig(
        name="fte_freq_16",
        encoding_type="fte",
        n_periods=16,
        loss_type="supcon",
        use_augmentation=True
    ),

    "fte_freq_32": ExperimentConfig(
        name="fte_freq_32",
        encoding_type="fte",
        n_periods=32,
        loss_type="supcon",
        use_augmentation=True
    ),

    "fte_freq_64": ExperimentConfig(
        name="fte_freq_64",
        encoding_type="fte",
        n_periods=64,
        loss_type="supcon",
        use_augmentation=True
    ),
    "fte_freq_128": ExperimentConfig(
        name="fte_freq_128",
        encoding_type="fte",
        n_periods=128,
        loss_type="supcon",
        use_augmentation=True
    )
}

# =============================================================================
# REVIEWER CONCERN #3: Factorial Ablation
# =============================================================================
# Address: "no full factorial ablation separating encoding/loss/augmentation"
#          "doesn't isolate its effect"
#          "how much gain is from LTE vs loss choice vs augmentation"

# Full Factorial: 3 encodings × 2 losses × 2 augmentation = 12 experiments

FACTORIAL_EXPERIMENTS = {
    # ==================== CLIP Encoding ====================
    "clip_supcon_aug": ExperimentConfig(
        name="clip_supcon_aug",
        encoding_type="clip",
        n_periods=16,  # Not used for CLIP
        loss_type="supcon",
        use_augmentation=True
    ),

    "clip_supcon_noaug": ExperimentConfig(
        name="clip_supcon_noaug",
        encoding_type="clip",
        n_periods=16,
        loss_type="supcon",
        use_augmentation=False
    ),

    "clip_arcface_aug": ExperimentConfig(
        name="clip_arcface_aug",
        encoding_type="clip",
        n_periods=16,
        loss_type="arcface",
        use_augmentation=True
    ),

    "clip_arcface_noaug": ExperimentConfig(
        name="clip_arcface_noaug",
        encoding_type="clip",
        n_periods=16,
        loss_type="arcface",
        use_augmentation=False
    ),

    # ==================== FTE Encoding ====================
    "fte_supcon_aug": ExperimentConfig(
        name="fte_supcon_aug",
        encoding_type="fte",
        n_periods=16,
        loss_type="supcon",
        use_augmentation=True
    ),

    "fte_supcon_noaug": ExperimentConfig(
        name="fte_supcon_noaug",
        encoding_type="fte",
        n_periods=16,
        loss_type="supcon",
        use_augmentation=False
    ),

    "fte_arcface_aug": ExperimentConfig(
        name="fte_arcface_aug",
        encoding_type="fte",
        n_periods=16,
        loss_type="arcface",
        use_augmentation=True
    ),

    "fte_arcface_noaug": ExperimentConfig(
        name="fte_arcface_noaug",
        encoding_type="fte",
        n_periods=16,
        loss_type="arcface",
        use_augmentation=False
    ),

    # ==================== LTE Encoding ====================
    "lte_supcon_aug": ExperimentConfig(
        name="lte_supcon_aug",
        encoding_type="lte",
        n_periods=16,
        loss_type="supcon",
        use_augmentation=True
    ),

    "lte_supcon_noaug": ExperimentConfig(
        name="lte_supcon_noaug",
        encoding_type="lte",
        n_periods=16,
        loss_type="supcon",
        use_augmentation=False
    ),

    "lte_arcface_aug": ExperimentConfig(
        name="lte_arcface_aug",
        encoding_type="lte",
        n_periods=16,
        loss_type="arcface",
        use_augmentation=True
    ),

    "lte_arcface_noaug": ExperimentConfig(
        name="lte_arcface_noaug",
        encoding_type="lte",
        n_periods=16,
        loss_type="arcface",
        use_augmentation=False
    ),
}

# =============================================================================
# ADDITIONAL: Loss Function Comparison
# =============================================================================
# Address: "SupCon different from ArcFace - no comment"
#          "ArcFace seems worse - why?"

LOSS_EXPERIMENTS = {
    "lte16_supcon": ExperimentConfig(
        name="lte16_supcon",
        encoding_type="lte",
        n_periods=16,
        loss_type="supcon",
        use_augmentation=True
    ),

    "lte16_arcface": ExperimentConfig(
        name="lte16_arcface",
        encoding_type="lte",
        n_periods=16,
        loss_type="arcface",
        use_augmentation=True
    ),

    "lte16_triplet": ExperimentConfig(
        name="lte16_triplet",
        encoding_type="lte",
        n_periods=16,
        loss_type="triplet",
        use_augmentation=True
    ),
}