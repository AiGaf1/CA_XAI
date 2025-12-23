import logging
import torch
from torch.export import Dim

from models.Litmodel import KeystrokeLitModel
from models.CNN import LearnPeriodsKeyEmb
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
import conf
import os

def setup_logger(name: str = None, level=logging.INFO) -> logging.Logger:
    """Set up and return a logger instance with a consistent format."""
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
        logger.propagate = False

    return logger
#
# def log_onnx_model(model: torch.nn.Module, logger: WandbLogger):
#

def export_to_onnx(model_checkpoint: ModelCheckpoint, wandb_logger: WandbLogger, periods: dict,  use_projector: bool):
    # After creating your model
    hold = torch.randn(conf.batches_per_epoch_train, conf.sequence_length, 1)
    flight = torch.randn(conf.batches_per_epoch_train, conf.sequence_length, 1)
    keys = torch.randint(0, 256, (conf.batches_per_epoch_train, conf.sequence_length, 1))  # Integer indices for embedding

    dummy_input = torch.cat([hold, flight, keys.long()], dim=-1)  # Shape: (1, 128, 3)
    model = LearnPeriodsKeyEmb(periods_dict=periods, use_projector=use_projector)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dummy_input = dummy_input.to(device)

    best_model = KeystrokeLitModel.load_from_checkpoint(
        model_checkpoint.best_model_path,
        model=model,
        strict=False
    )
    best_model.eval()

    float_score = float(model_checkpoint.best_model_score)
    rounded_score = round(float_score, 2)
    onnx_path = f"model_{rounded_score}.onnx"

    # Define dynamic batch dimension
    batch = Dim("batch_size", min=1, max=1024)
    dynamic_shapes = {"input": {0: batch}}  # {0: batch} means first dim is dynamic

    with torch.no_grad():
        torch.onnx.export(
            best_model,
            (dummy_input,),
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            # dynamic_shapes=dynamic_shapes
        )

    run_id = model_checkpoint.best_model_path.split(os.sep)[-3]

    artifact = wandb.Artifact(
        name=f"model-onnx-{run_id}-{rounded_score}",
        type="model",
        description="ONNX exported model"
    )

    artifact.add_file(local_path=onnx_path)

    wandb_logger.experiment.log_artifact(artifact)
    artifact.wait()