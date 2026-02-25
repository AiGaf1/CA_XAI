import torch
from pytorch_metric_learning.miners import BaseMiner
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from data.Aalto.metrics import compute_eer
from typing import Any, Dict, Tuple, Optional


class KeystrokeLitModel(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Optional[torch.nn.Module],
        t_0: float = 2500,
        lr: float = 1e-3,
        miner: Optional[BaseMiner] = None,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.miner = miner

        for name, module in self.named_modules():
            module.name = name
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model(x, mask)

    def _step_common(
        self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor],
            torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], stage: str
    ) -> Dict[str, Any]:
        """
        batch = ((x1_features, x1_keys), (x2_features, x2_keys)), labels, (u1, u2)
        """
        (x1, mask1), (x2, mask2), labels, (_, _), (u1, u2) = batch

        z1 = self.model(x1, mask1)
        z2 = self.model(x2, mask2)

        embeddings = torch.cat([z1, z2], dim=0)
        user_idx = torch.cat([u1, u2], dim=0)

        loss = self.loss_fn(embeddings, user_idx)

        with torch.inference_mode():
            eer = compute_eer(z1.detach().float().cpu(), z2.detach().float().cpu(), labels.detach().float().cpu())

        # Logging
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}/eer", eer, prog_bar=True, on_epoch=True, on_step=False)

        return loss

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

    def predict_step(self, batch, batch_idx):
        fix_sessions, masks, session_ids = batch

        # Get embeddings from model (pass mask if your model uses it)
        embeddings = self.forward(fix_sessions, mask=masks)  # adjust based on your model's forward signature
        # OR if model doesn't use mask:
        # embeddings = self.forward(fix_sessions)

        # Return list of (session_id, embedding) tuples
        return [(sid, emb.cpu()) for sid, emb in zip(session_ids, embeddings)]

    def on_train_start(self):
        # Attach model/grad logging in W&B
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.watch(self.model, log="all",
                                         log_freq=100, log_graph=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.95, 0.999)
        )

        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.hparams.t_0,
            eta_min=self.hparams.lr / 100
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }