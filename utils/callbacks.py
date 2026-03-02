import os
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


def create_callbacks(scenario: str, run_dir: str = None) -> list[pl.Callback]:
    """Create training callbacks for checkpointing and monitoring."""
    dirpath = os.path.join(run_dir, "checkpoints") if run_dir else None
    checkpoint_cb = ModelCheckpoint(
        dirpath=dirpath,
        monitor="val/eer",
        mode="min",
        filename=f'{scenario}' + "-{epoch:02d}-{val/eer:.2f}",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False
    )

    lr_monitor = LearningRateMonitor(logging_interval=None)
    # validation_silent_bar = ValidationSilentProgressBar()

    return [checkpoint_cb, lr_monitor]  # , validation_silent_bar]
