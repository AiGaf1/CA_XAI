import torch.nn as nn
import config as conf
from models.transformer import KeystrokeTransformer
from models.cnn import KeystrokeCNN
from models.lstm import KeystrokeLSTM


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
            use_pos_enc=cfg.use_pos_enc,
            use_mste=config.use_mste,
        )
    elif config.model_type == "cnn":
        cfg: conf.CNNConfig = config.model
        return KeystrokeCNN(
            periods_dict=periods_dict,
            hidden_size=cfg.hidden_size,
            output_size=config.output_size,
            sequence_length=config.window_size,
            use_projector=config.use_projector,
            n_periods=cfg.n_periods,
            use_mste=config.use_mste,
        )
    elif config.model_type == "lstm":
        cfg: conf.LSTMConfig = config.model
        return KeystrokeLSTM(
            periods_dict=periods_dict,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            use_projector=config.use_projector,
            n_periods=cfg.n_periods,
            use_mste=config.use_mste,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
    else:
        raise ValueError(
            f"Unknown model_type: '{config.model_type}'. Expected 'transformer', 'cnn', or 'lstm'."
        )
