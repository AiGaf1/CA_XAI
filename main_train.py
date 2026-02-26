# 1. Standard Library
import os
# 2. Third-Party Libraries
import numpy as np
import pytorch_lightning as pl
import wandb
from dotenv import load_dotenv
from pytorch_lightning.loggers import WandbLogger
from torch import nn
import torch
# 3. Local Modules (Project Specific)
import conf
from data.Aalto.dataset import KeystrokeDataModule
from models.Litmodel import KeystrokeLitModel
from models.factory import build_model
from utils.callbacks import create_callbacks
from utils.losses import build_loss
from utils.tools import (
    export_to_onnx,
    setup_logger,
    setup_wandb_logging
)

logger = setup_logger("main")



# class PositiveOnlyReducer(BaseReducer):
#     def element_reduction(self, losses, *args, **kwargs):
#         return torch.sum(losses[losses > 0])

#     def triplet_reduction(self, losses, loss_indices, embeddings, labels, **kwargs):
#         positive_mask = losses > 0
#         if positive_mask.sum() == 0:
#             return torch.tensor(0.0, device=losses.device, dtype=losses.dtype)
#         return torch.sum(losses[positive_mask])

# class TripletLoss(nn.Module):
#     def __init__(self, margin=0.25):
#         super().__init__()
#         self.loss_fn = losses.TripletMarginLoss(
#             margin=margin,
#             distance=distances.CosineSimilarity(),
#             reducer=PositiveOnlyReducer(),  # ← Your custom reducer!
#             swap=True
#         )
#     def forward(self, embeddings, labels):
#         return self.loss_fn(embeddings, labels)

def initialize_environment(config: conf.ExperimentConfig) -> None:
    """Initialize training environment with reproducibility and hardware optimization."""
    pl.seed_everything(config.seed, workers=True, verbose=False)
    load_dotenv()

    api_key = os.getenv("WANDB_API_KEY")  # Fixed typo from "WAND_API_KEY"
    if api_key:
        wandb.login(key=api_key)
    else:
        logger.warning("WANDB_API_KEY not found in environment variables.")

    # Optimize for modern GPUs (Ampere and later)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    logger.info(
        f"Environment initialized | Seed: {config.seed} | Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

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
        log_every_n_steps=100,
        num_sanity_val_steps=1
    )

def run_predictions(
        trainer: pl.Trainer,
        model: KeystrokeLitModel,
        checkpoint_path: str = "best"
) -> dict[str, torch.Tensor]:
    """Run predictions and collect embeddings."""
    logger.info("Running predictions...")

    predictions = trainer.predict(
        model,
        ckpt_path=checkpoint_path
    )

    # Aggregate embeddings from all batches
    embeddings = {}
    for batch_predictions in predictions:
        for session_id, embedding in batch_predictions:
            embeddings[session_id] = embedding

    logger.info(f"Collected {len(embeddings)} session embeddings")
    return embeddings

def run_experiment(config: conf.ExperimentConfig) -> None:
    initialize_environment(config)

    # 1. Data & Model Setup
    dm = create_data_module(config.file_path, config.predict_file_path, config)

    loss_fn = build_loss(config)
    nn_model = build_model(config, dm.min_max)
    nn_model = torch.compile(nn_model, mode='default')
    lit_model = KeystrokeLitModel(nn_model, loss_fn, t_0=config.t_0, lr=config.lr)

    # 3. Training Infrastructure
    wandb_logger, run_dir = setup_wandb_logging(config=config, model_name=nn_model.__class__.__name__)
    callbacks = create_callbacks(config.scenario, run_dir)
    trainer = create_trainer(config.epochs, wandb_logger, callbacks)

    # 4. Execution
    trainer.fit(lit_model, datamodule=dm)

    # 5. Post-Training: Exporting the BEST model
    checkpoint_cb = callbacks[0]
    best_model_path = checkpoint_cb.best_model_path
    logger.info(f"Loading best model for export: {best_model_path}")
    export_to_onnx(config, best_model_path, wandb_logger, nn_model)
    wandb.finish()

    # # Run predictions
    # ckpt_path = "Keystroke-XAI/20251227_0330/checkpoints/mobile-769-1.49.ckpt"
    # embeddings = run_predictions(trainer, lit_model, dm, ckpt_path)
    # comparisons = load_comparisons(conf.scenario, logger)
    # #
    # distances = compute_distances4comps(embeddings, comparisons, metric="euclidean")
    # similarities = normalize_distances(distances) # No need for second normalization
    # #
    # # save_predictions(similarities, conf.scenario, logger)
    #
    # # visualize_activations(nn_model, dm)

if __name__ == "__main__":
    config = conf.ExperimentConfig(scenario='desktop')
    run_experiment(config)