from typing import Any, Dict, Tuple
from dotenv import load_dotenv
import os
from datetime import datetime
import numpy as np
import wandb
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_metric_learning.losses import SupConLoss


from lejepa.univariate import EppsPulley
from lejepa.multivariate import SlicingUnivariateTest
from models.CNN import LearnPeriodsKeyEmb, norm_embeddings
from data.dataset import KeystrokeDataModule
from data.metrics import compute_eer
from utils.callbacks import ValidationSilentProgressBar
from utils.tools import setup_logger
import conf

logger = setup_logger("main")

sequence_length = 128
embedding_size = 512

lr_scheduler_T_max = 3000 #steps
# lr_scheduler_T_max = loss_conf['batches_per_epoch_train'] * 20 # 20 epochs for full circle
PROJECT = "Keystroke-XAI"

def init_launch():
    pl.seed_everything(conf.seed, workers=True)
    load_dotenv()
    wandb.login(key=os.getenv("WAND_API_KEY"))
    torch.set_float32_matmul_precision('high')

class KeystrokeLitModel(pl.LightningModule):
    def __init__(self, dm: KeystrokeDataModule, model: torch.nn.Module, lr: float = 1e-2,
                 lambda_sigreg: float = 0.05, sigreg_slices: int = 1024):
        super().__init__()
        self.save_hyperparameters(ignore=['dm', 'model'])
        self.model = model
        self.dm = dm
        self.supcon = SupConLoss()

        univariate_test = EppsPulley()
        self.sigreg = SlicingUnivariateTest(
            univariate_test=univariate_test,
            num_slices=sigreg_slices,
        )

    def forward(self, batch_x: tuple) -> torch.Tensor:
        return self.model(batch_x.float())

    def encode_pair(self, x1, x2):
        """Encode two views and return raw + L2-normalized versions"""
        z1_raw = self(x1)
        z2_raw = self(x2)

        z1_norm = norm_embeddings(z1_raw)
        z2_norm = norm_embeddings(z2_raw)

        return {
            "raw": torch.cat([z1_raw, z2_raw], dim=0),
            "norm": torch.cat([z1_norm, z2_norm], dim=0),
        }

    def _step_common(
        self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor],
            torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], stage: str
    ) -> Dict[str, Any]:
        """
        batch = ((x1_features, x1_keys), (x2_features, x2_keys)), labels, (u1, u2)
        """
        (x1, x2), labels, (u1, u2) = batch

        # Encode both views
        encodings = self.encode_pair(x1, x2)
        z_raw_all = encodings["raw"]  # [2B, D] – for SIGReg
        z_norm_all = encodings["norm"]  # [2B, D] – for SupCon & EER
        user_idx = torch.cat([u1, u2], dim=0)

        # Losses
        supcon_loss = self.supcon(z_norm_all, user_idx)
        sigreg_loss = self.sigreg(z_raw_all)
        total_loss = (1 - self.hparams.lambda_sigreg) * supcon_loss + self.hparams.lambda_sigreg * sigreg_loss

        # EER on normalized embeddings (standard for keystroke auth)
        z1_norm, z2_norm = z_norm_all.chunk(2)
        with torch.inference_mode():
            eer = compute_eer(z1_norm.detach().cpu(), z2_norm.detach().cpu(), labels.detach().cpu())

        # Logging
        self.log(f"{stage}/supcon_loss", supcon_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}/sigreg_loss", sigreg_loss, prog_bar=(stage == "train"), on_epoch=True, on_step=False)
        self.log(f"{stage}/total_loss", total_loss, prog_bar=True, on_epoch=True,  on_step=False)
        self.log(f"{stage}/eer", eer, prog_bar=True, on_epoch=True, on_step=False)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._step_common(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step_common(batch, "val")

    # def test_step(self, batch, batch_idx):
    #     out = self._step_common(batch)
    #     test_loss = contrastive_loss(out["z1"], out["z2"], out["labels"].float())
    #     self.log("test/loss", test_loss, prog_bar=True, on_epoch=True)
    #     self.log("test/eer", out["eer"], prog_bar=True, on_epoch=True)
    #     return {"test_loss": test_loss.detach()}

    def on_train_start(self):
        # Attach model/grad logging in W&B
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.watch(self.model, log="all", log_freq=100)

    def configure_optimizers(self):
        lr = self.hparams.lr
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            betas=(0.95, 0.999)
        )

        cosine_scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=lr_scheduler_T_max, eta_min=lr/10000)
        return [optimizer], [
            {"scheduler": cosine_scheduler, "interval": "step", "frequency": 1}
        ]

def run_experiment(file_path: str):
    FULL_NAME = f'{conf.N_PERIODS}_{conf.EPOCHS}_{conf.scenario}'
    init_launch()
    logger.info("Data Loading...")
    data = np.load(file_path, allow_pickle=True).item()

    dm = KeystrokeDataModule(
        data=data,
        sequence_length=sequence_length,
        samples_per_batch_train=conf.samples_per_batch_train,
        samples_per_batch_val=conf.samples_per_batch_val,
        batches_per_epoch_train=conf.batches_per_epoch_train,
        batches_per_epoch_val=conf.batches_per_epoch_val,
        train_val_division=conf.train_val_division,
        augment=True,
        seed=conf.seed,
        )

    dm.setup(None)

    nn_model = LearnPeriodsKeyEmb(periods_dict=dm.init_periods)

    # -----------------------------
    # Logging & Callbacks
    # -----------------------------
    tags = [
        f"scenario_{conf.scenario}",
        f"embedding_{embedding_size}",
        f"seqlen_{sequence_length}",
        f"epochs_{conf.EPOCHS}",
    ]
    version = datetime.now().strftime("%Y%m%d_%H%M")
    wandb_logger = WandbLogger(project=PROJECT, name=FULL_NAME, version=version,
                               log_model=True, tags=tags)

    checkpoint_cb = ModelCheckpoint(
        monitor="val/eer",
        mode="min",
        filename= f'{conf.scenario}' + "-{epoch:02d}-{val/eer:.2f}",
        save_top_k=2,
        save_last=True,
        auto_insert_metric_name=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval=None)
    validation_silent_bar = ValidationSilentProgressBar()

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {accelerator}")

    trainer = pl.Trainer(
        max_epochs=conf.EPOCHS,
        logger=wandb_logger,
        callbacks=[checkpoint_cb, lr_monitor, validation_silent_bar],
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        accelerator=accelerator,
        devices=1,
        deterministic=True,
        log_every_n_steps=10,
        num_sanity_val_steps=2,
    )
    # -----------------------------
    # Fit
    # -----------------------------
    full_model = KeystrokeLitModel(dm, model=nn_model, lr=1e-2)
    trainer.fit(full_model, datamodule=dm)

    best_ckpt_path = checkpoint_cb.best_model_path
    print(f"Best checkpoint: {best_ckpt_path}")

if __name__ == "__main__":
    file_path = f'data/{conf.scenario}/{conf.scenario}_dev_set.npy'
    run_experiment(file_path)
    wandb.finish()
