# 1. Standard Library
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")
warnings.filterwarnings("ignore", message=".*dynamic_axes.*dynamo.*")
warnings.filterwarnings("ignore", message=".*dtype.*align.*", module="numpy")
warnings.filterwarnings("ignore", message=".*wandb run already in progress.*")
warnings.filterwarnings("ignore", message=".*bf16-mixed is not supported by the model summary.*")

import logging
logging.getLogger("torch._inductor.autotune_process").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)

# 2. Third-Party Libraries
import numpy as np
import pytorch_lightning as pl
import wandb
from dotenv import load_dotenv
from pytorch_lightning.loggers import WandbLogger
from torch import nn
import torch
# 3. Local Modules (Project Specific)
import config as conf
from data.Aalto.dataset import KeystrokeDataModule
from models.lit_model import KeystrokeLitModel
from models.factory import build_model
from utils.callbacks import create_callbacks
from utils.losses import build_loss
from utils.tools import (
    export_to_onnx,
    load_comparisons,
    save_predictions,
    setup_logger,
    setup_wandb_logging
)
from utils.metrics import compute_distances4comps, normalize_distances
from utils.sweep import run_sweep_agent

logger = setup_logger("train")

def wandb_login() -> None:
    load_dotenv()
    os.makedirs("outputs", exist_ok=True)
    os.environ["WANDB_DIR"] = os.path.abspath("outputs")
    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    else:
        logger.warning("WANDB_API_KEY not found in environment variables.")


def initialize_environment(config: conf.ExperimentConfig) -> None:
    pl.seed_everything(config.seed, workers=True, verbose=False)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
    logger.info(f"Environment initialized | Seed: {config.seed} | Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

def create_data_module(
        file_path: str,
        predict_file_path: str = None,
        config: conf.ExperimentConfig = None,
) -> KeystrokeDataModule:
    """Create and setup the data module for training and prediction."""
    logger.info(f"Loading data from {file_path}")
    raw_data = np.load(file_path, allow_pickle=True).item()

    dm = KeystrokeDataModule(
        raw_data=raw_data,
        predict_file_path=predict_file_path,
        window_size=config.window_size, # conf.window_size, min_session_length
        samples_per_batch_train=config.samples_per_batch_train,
        samples_per_batch_val=config.samples_per_batch_val,
        batches_per_epoch_train=config.batches_per_epoch_train,
        batches_per_epoch_val=config.batches_per_epoch_val,
        train_val_division=config.train_split,
        seed=config.seed,
        min_session_length=config.min_session_length,
        min_sessions_per_user=config.min_sessions_per_user
    )
    dm.setup(None)
    return dm

def create_trainer(
        epochs: int,
        wandb_logger: WandbLogger = None,
        callbacks: list[pl.Callback] = None,
) -> pl.Trainer:
    """Create PyTorch Lightning trainer with optimal settings."""
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {accelerator}")

    return pl.Trainer(
        max_epochs=epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        accelerator=accelerator,
        devices=1,
        deterministic=True,
        log_every_n_steps=256, #per epoch
        num_sanity_val_steps=1
    )

def run_predictions(
        trainer: pl.Trainer,
        model: KeystrokeLitModel,
        dm: KeystrokeDataModule,
        checkpoint_path: str = "best"
) -> dict[str, torch.Tensor]:
    """Run predictions and collect embeddings."""
    logger.info("Running predictions...")

    predictions = trainer.predict(
        model,
        datamodule=dm,
        ckpt_path=checkpoint_path
    )

    # Aggregate embeddings from all batches
    embeddings = {}
    for batch_predictions in predictions:
        for session_id, embedding in batch_predictions:
            embeddings[session_id] = embedding

    logger.info(f"Collected {len(embeddings)} session embeddings")
    return embeddings

def run_experiment(config: conf.ExperimentConfig, sweep_run_id: str = None) -> None:
    initialize_environment(config)

    # 1. Data & Model Setup
    dm = create_data_module(config.file_path, config.predict_file_path, config)

    loss_fn = build_loss(config)
    nn_model = build_model(config, dm.min_max)
    lit_model = KeystrokeLitModel(nn_model, loss_fn, t_0=config.t_0, lr=config.lr)

    # 3. Training Infrastructure
    wandb_logger, run_dir = setup_wandb_logging(config=config, model_name=nn_model.__class__.__name__, run_id=sweep_run_id)
    callbacks = create_callbacks(config.scenario, run_dir)
    trainer = create_trainer(config.epochs, wandb_logger, callbacks)

    # 4. Execution
    trainer.fit(lit_model, datamodule=dm)

    # 5. Post-Training: Exporting the BEST model
    checkpoint_cb = callbacks[0]
    best_model_path = checkpoint_cb.best_model_path
    logger.info(f"Loading best model for export: {best_model_path}")
    export_to_onnx(config, best_model_path, wandb_logger, nn_model)

    # # Run predictions
    # ckpt_path = "Keystroke-XAI/20251227_0330/checkpoints/mobile-769-1.49.ckpt"
    embeddings = run_predictions(trainer, lit_model, dm, best_model_path)
    comparisons = load_comparisons(config.scenario, logger)
    distances = compute_distances4comps(embeddings, comparisons, metric="euclidean")
    similarities = normalize_distances(distances)
    save_predictions(similarities, config.scenario, run_dir, logger)
    # visualize_activations(nn_model, dm)

    wandb.finish()


if __name__ == "__main__":
    wandb_login()
    initialize_environment(conf.SWEEP_BASE)
    run_sweep_agent(run_experiment)
