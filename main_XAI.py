from dotenv import load_dotenv
import os
import numpy as np
from datetime import datetime

import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.tuner import Tuner
import matplotlib.pyplot as plt
import math
import seaborn as sns
from torch import nn
import h5py



from data.dataset import KeystrokeDataModule
from utils.tools import setup_logger
from utils.visualization import visualize_keystrokes
from models.LTE import  LTEOrig, MultiplyFreqs, SinCos, ScaledSigmoid
import conf


logger = setup_logger("main")

def init_launch():
    pl.seed_everything(conf.seed, workers=True)
    load_dotenv()
    wandb.login(key=os.getenv("WAND_API_KEY"))
    torch.set_float32_matmul_precision('high')

def run_experiment(file_path: str):
    FULL_NAME = f'{conf.epochs}_{conf.scenario}'
    init_launch()
    logger.info("Data Loading...")
    data = np.load(file_path, allow_pickle=True).item()

    dm = KeystrokeDataModule(
        data=data,
        sequence_length=conf.sequence_length,
        samples_per_batch_train=conf.samples_per_batch_train,
        samples_per_batch_val=conf.samples_per_batch_val,
        batches_per_epoch_train=conf.batches_per_epoch_train,
        batches_per_epoch_val=conf.batches_per_epoch_val,
        train_val_division=conf.train_val_division,
        augment=True,
        seed=conf.seed,
        )

    dm.setup(None)

    small_loader = dm.train_dataloader()
    (x1, x2), labels, (u1, u2) = next(iter(small_loader))
    print((x1[0], x2[0]), labels[0], (u1[0], u2[0]))
    visualize_keystrokes(x1[0])
    visualize_keystrokes(x1[1])
   #  nn_model = LearnPeriodsKeyEmb(periods_dict=dm.init_periods)
   # # -----------------------------
   #  # Logging & Callbacks
   #  # -----------------------------
   #  tags = [
   #      f"scenario_{conf.scenario}",
   #      f"embedding_{conf.embedding_size}",
   #      f"seqlen_{conf.sequence_length}",
   #      f"epochs_{conf.epochs}",
   #      f"trigperiods_{conf.N_PERIODS}"
   #  ]
   #  version = datetime.now().strftime("%Y%m%d_%H%M")
   #  wandb_logger = WandbLogger(project=conf.project, name=FULL_NAME, version=version,
   #                             log_model=True, tags=tags)
   #
   #
   #  accelerator = "gpu" if torch.cuda.is_available() else "cpu"
   #  logger.info(f"Device: {accelerator}")
   #
   #  # -----------------------------
   #  # Fit
   #  # -----------------------------
   #  full_model = KeystrokeLitModel(nn_model, lr=1e-3)
   #  trainer.fit(full_model, datamodule=dm)
   #  export_to_onnx(checkpoint_cb, wandb_logger, dm.init_periods)
   #  visualize_activations(nn_model, dm)

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
                                                            nn.BatchNorm2d, nn.LayerNorm, LTEOrig)):
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

if __name__ == "__main__":
    file_path = f'data/{conf.scenario}/{conf.scenario}_dev_set.h5'
    run_experiment(file_path)
    wandb.finish()