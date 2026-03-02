"""
Ablation Study Configuration
Define different experimental configurations here
"""
from dataclasses import dataclass, field
from typing import Union


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

    # Model architecture
    model_type: str = "transformer"  # "transformer" | "cnn"
    model: Union[TransformerConfig, CNNConfig] = field(default_factory=TransformerConfig)
    hidden_size: int = 128
    output_size: int = 512
    use_projector: bool = False

    # Loss function
    loss_type: str = "supcon"  # "supcon" | "arcface" | "triplet"
    arcface_num_classes: int = 100

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

    # Data path
    scenario: str = 'desktop'
    file_path: str = ''
    predict_file_path: str = ''

    # Other
    seed: int = 42

    def __post_init__(self):
        if not self.file_path:
            self.file_path = f'data/Aalto/raw/{self.scenario}/{self.scenario}_dev_set.npy'
        if not self.predict_file_path:
            self.predict_file_path = f'data/{self.scenario}/{self.scenario}_test_sessions.npy'


# ---------------------------------------------------------------------------
# Sweep configuration — edit these to define your ablation study.
# Keys in SWEEP_GRID map to ExperimentConfig fields.
# Keys prefixed with "model." set fields on the nested model sub-config.
# ---------------------------------------------------------------------------
SWEEP_GRID = {
    "scenario":         ["desktop", "mobile"],
    "loss_type":        ["supcon", "arcface"],
    "model.num_layers": [4, 6],
    "lr":               [1e-3, 5e-4],
}

# Combinations to skip — each entry is a dict of field:value pairs that must all match.
SWEEP_EXCLUDE = [
    {"scenario": "desktop", "loss_type": "arcface"},
    {"scenario": "mobile",  "loss_type": "supcon"},
]

# Only override what differs from ExperimentConfig defaults.
SWEEP_BASE = ExperimentConfig(
    model=TransformerConfig(num_heads=2),
)
