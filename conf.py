from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str = "transformer"  # "cnn" or "transformer"
    sequence_length: int = 64
    use_projector: bool = True
    n_periods: int = 64
    lr: float = 1e-3


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_train: int = 32
    batch_val: int = 32
    seed: int = 42


@dataclass
class ExperimentConfig:
    scenario: str = "mobile"
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()



#----Project-variables------
scenario = 'mobile'
project = "Keystroke-XAI"
seed = 40
#----Data-Variables------
train_val_division = 0.80
#----Loss-Variables------
batches_per_epoch_train = 256
batches_per_epoch_val = 16
samples_per_batch_train = 512
samples_per_batch_val = 512
#----Model-Variables------
N_PERIODS = 64
sequence_length = 64
embedding_size = 128
use_projector = False
#----Optimization-Variables------
lr_scheduler_T_max = 2500
epochs = 1000
