from dotenv import load_dotenv
import os
import numpy as np

import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import math
import seaborn as sns
from torch import nn

from data.dataset import KeystrokeDataModule
from utils.tools import setup_logger
from models.Litmodel import KeystrokeLitModel
from utils.callbacks import create_callbacks
from models.CNN import CNN_LTE, norm_embeddings
from models.transformer import Transformer_LTE
from models.LTE import  LearnableFourierFeatures
from utils.tools import export_to_onnx, setup_wandb_logging, save_predictions, load_comparisons
import conf

logger = setup_logger("main")

def initialize_environment() -> None:
    """Initialize training environment with reproducibility settings."""
    pl.seed_everything(conf.seed, workers=True, verbose=False)
    load_dotenv()
    wandb.login(key=os.getenv("WAND_API_KEY"))
    torch.set_float32_matmul_precision('high')

    logger.info(f"Environment initialized with seed: {conf.seed}")

def create_data_module(
        file_path: str,
        predict_file_path: str = None,
        min_session_length: int = 5,
        sequence_length: int = conf.sequence_length,
        min_sessions_per_user: int = 2
) -> KeystrokeDataModule:
    """Create and setup the data module for training and prediction."""
    logger.info("Loading data...")
    raw_data = np.load(file_path, allow_pickle=True).item()

    dm = KeystrokeDataModule(
        raw_data=raw_data,
        predict_file_path=predict_file_path,
        sequence_length=sequence_length, # conf.sequence_length, min_session_length
        samples_per_batch_train=conf.samples_per_batch_train,
        samples_per_batch_val=conf.samples_per_batch_val,
        batches_per_epoch_train=conf.batches_per_epoch_train,
        batches_per_epoch_val=conf.batches_per_epoch_val,
        train_val_division=conf.train_val_division,
        augment=True,
        seed=conf.seed,
        min_session_length=min_session_length,
        min_sessions_per_user=min_sessions_per_user
    )
    dm.setup(None)

    logger.info(f"Data module created with {dm.num_train_users} training users")
    return dm

def create_trainer(
        wandb_logger: WandbLogger = None,
        callbacks: list[pl.Callback] = None
) -> pl.Trainer:
    """Create PyTorch Lightning trainer with optimal settings."""
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {accelerator}")

    return pl.Trainer(
        max_epochs=conf.epochs,
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
        datamodule: KeystrokeDataModule,
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


def visualize_activations(net, datamodule, color="C0"):
    """
    Visualize activations throughout the network by registering forward hooks.
    Adapted from the PyTorch Lightning UvA Deep Learning course.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net.eval()
    activations = {}
    # Hook function to capture activations
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks for layers with parameters or specific activation types
    hooks = []
    for name, module in net.named_modules():
        if hasattr(module, 'weight') or isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Tanh,
                                                            nn.Sigmoid, nn.GELU, nn.BatchNorm1d,
                                                            nn.BatchNorm2d, nn.LayerNorm, LearnableFourierFeatures)):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # Get a batch of data and run forward pass
    small_loader = datamodule.train_dataloader()
    (x1, x2), labels, (u1, u2) = next(iter(small_loader))

    with torch.no_grad():
        _ = net(x1.float().to(device))

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Process activations for plotting
    processed_activations = {name: activation.view(-1).cpu().numpy()
                            for name, activation in activations.items()}

    # Create subplot grid
    columns = 3
    rows = math.ceil(len(processed_activations) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 4, rows * 3))
    axes = np.atleast_2d(axes)

    # Plot each layer's activations
    for idx, (name, activation_np) in enumerate(processed_activations.items()):
        row, col = idx // columns, idx % columns
        ax = axes[row, col]

        sns.histplot(data=activation_np, bins=50, ax=ax, color=color, kde=True, stat="density")
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.3)

        display_name = name.replace('model.', '')
        module_type = type(dict(net.named_modules())[name]).__name__
        ax.set_title(f"{display_name}\n({module_type})", fontsize=10)

        stats_text = f"μ={activation_np.mean():.2f}, σ={activation_np.std():.2f}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=8)

    # Turn off unused subplots
    for idx in range(len(processed_activations), rows * columns):
        row, col = idx // columns, idx % columns
        axes[row, col].axis('off')

    fig.suptitle("Activation distributions", fontsize=14)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.4, wspace=0.3)
    plt.savefig("activations_visualization.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return processed_activations

def compute_distances4comps(embeddings, comps, metric="euclidean"):
    distances = {}
    for comp in comps:
        e1, e2 = norm_embeddings(embeddings[comp[0]]), norm_embeddings(embeddings[comp[1]])
        distance = compute_distance(e1, e2, metric)
        distances[str(comp)] = distance
    return distances

def compute_distance(e1, e2, metric):
    if metric == "euclidean":
        distance = nn.functional.pairwise_distance(e1, e2).item()
    elif metric == "cosine":
        distance = nn.functional.cosine_similarity(e1, e2).item()
    else:
        raise ValueError(f"Unknown metric {metric}")
    return distance

def normalize_distances(distances):
    """
    Convert distances to similarity in [0, 1]
    """
    distances_list = torch.tensor(list(distances.values()), dtype=torch.float32)
    max_dist = distances_list.max()
    similarities = 1 - distances_list / max_dist
    return similarities.tolist()

def run_experiment(file_path: str, predict_file_path: str):

    initialize_environment()
    # Setup data and model
    dm = create_data_module(file_path, predict_file_path)
    nn_model = Transformer_LTE(periods_dict=dm.min_max, use_projector=conf.use_projector,
                               sequence_length=conf.sequence_length)
    # Setup training
    wandb_logger, version = setup_wandb_logging(use_projector=conf.use_projector)
    callbacks = create_callbacks(conf.scenario)
    trainer = create_trainer(wandb_logger, callbacks)

    lit_model = KeystrokeLitModel(nn_model, lr=1e-3)
    trainer.fit(lit_model, datamodule=dm)
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"Best model path: {best_model_path}")

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
    checkpoint_cb = callbacks[0]
    best_model_path = checkpoint_cb.best_model_path
    print(f"Best model path: {best_model_path}")
    export_to_onnx(best_model_path, wandb_logger, nn_model)
    wandb.finish()
    # # visualize_activations(nn_model, dm)

if __name__ == "__main__":
    file_path = f'data/{conf.scenario}/{conf.scenario}_dev_set.npy'
    predict_file_path = f'data/{conf.scenario}/{conf.scenario}_test_sessions.npy'

    # n_periods = [4, 8, 16, 32, 64]
    #
    # for i in n_periods:
    conf.N_PERIODS = 64
    run_experiment(file_path, predict_file_path)