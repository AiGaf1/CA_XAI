import torch.nn as nn
import config as conf
from models.transformer import KeystrokeTransformer
from models.cnn import KeystrokeCNN


def build_model(config: conf.ExperimentConfig, periods_dict) -> nn.Module:
    """Instantiate the model specified by config.model_type."""
    if config.model_type == "transformer":
        cfg: conf.TransformerConfig = config.model
        return KeystrokeTransformer(
            periods_dict=periods_dict,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            window_size=config.window_size,
            use_projector=config.use_projector,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            ff_dim=cfg.ff_dim,
            dropout=cfg.dropout,
            n_periods=cfg.n_periods,
        )
    elif config.model_type == "cnn":
        cfg: conf.CNNConfig = config.model
        return KeystrokeCNN(
            periods_dict=periods_dict,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            sequence_length=config.window_size,
            use_projector=config.use_projector,
            n_periods=cfg.n_periods,
        )
    else:
        raise ValueError(
            f"Unknown model_type: '{config.model_type}'. Expected 'transformer' or 'cnn'."
        )
