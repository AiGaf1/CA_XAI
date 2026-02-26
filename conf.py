"""
Ablation Study Configuration
Define different experimental configurations here
"""
from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class TransformerConfig:
    """Transformer-specific hyperparameters"""
    num_layers: int = 4
    num_heads: int = 1
    ff_dim: int = 512
    dropout: float = 0.1
    n_periods: int = 16


@dataclass
class CNNConfig:
    """CNN-specific hyperparameters"""
    n_periods: int = 16


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
    model_type: str = "transformer"  # "transformer" | "cnn"
    model: Union[TransformerConfig, CNNConfig] = field(default_factory=TransformerConfig)
    encoding_type: str = "lte"  # "lte", "fte", "clip"
    hidden_size: int = 128
    output_size: int = 512
    use_projector: bool = False

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
    file_path: str = f'data/Aalto/raw/{scenario}/{scenario}_dev_set.npy'
    predict_file_path: str = f'data/{scenario}/{scenario}_test_sessions.npy'

    # Other
    seed: int = 42