from dataclasses import dataclass, field
from typing import Union


@dataclass
class TransformerConfig:
    num_layers: int = 4
    num_heads: int = 1
    ff_dim: int = 512
    dropout: float = 0.1
    n_periods: int = 16
    use_pos_enc: bool = True
    use_sigmoid: bool = True


@dataclass
class CNNConfig:
    n_periods: int = 16


@dataclass
class ExperimentConfig:
    name: str = "Keystroke-XAI"

    # Data
    window_size: int = 50
    min_session_length: int = 5
    min_sessions_per_user: int = 2
    train_split: float = 0.80

    # Model
    model_type: str = "transformer"  # "transformer" | "cnn"
    model: Union[TransformerConfig, CNNConfig] = field(default_factory=TransformerConfig)
    hidden_size: int = 128
    output_size: int = 512
    use_projector: bool = False

    # Loss
    loss_type: str = "supcon"  # "supcon" | "arcface" | "triplet"
    arcface_num_classes: int = 100

    # Training
    epochs: int = 1000
    lr: float = 1e-3
    t_0: int = 2500
    batches_per_epoch_train: int = 256
    batches_per_epoch_val: int = 16
    samples_per_batch_train: int = 512
    samples_per_batch_val: int = 512

    # Data path
    scenario: str = 'desktop'
    file_path: str = ''
    predict_file_path: str = ''

    seed: int = 42

    def __post_init__(self):
        if not self.file_path:
            self.file_path = f'data/Aalto/raw/{self.scenario}/{self.scenario}_dev_set.npy'
        if not self.predict_file_path:
            self.predict_file_path = f'data/Aalto/raw/{self.scenario}/{self.scenario}_test_sessions.npy'


# Base config — only override what differs from ExperimentConfig defaults.
SWEEP_BASE = ExperimentConfig(
    model=TransformerConfig(num_heads=2),
)

# W&B native sweep config — passed to run_sweep_agent() in train.py.
WANDB_SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "val/loss", "goal": "minimize"},
    "parameters": {
        # "scenario":          {"values": ["desktop"]},
        # "loss_type":         {"values": ["supcon"]},
        # "model.num_layers":  {"values": [4]},
        "model.use_pos_enc": {"values": [True, False]},
        "model.use_sigmoid": {"values": [True, False]},
    },
}
