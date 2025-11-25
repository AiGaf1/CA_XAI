from pytorch_lightning.callbacks import TQDMProgressBar
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl


class ValidationSilentProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.disable = True  # Disable only validation progress bar
        return bar


class ReduceCosineMaxOnPlateau(pl.Callback):
    def __init__(self, monitor='val/eer', mode='min', factor=0.1, patience=99, verbose=True):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        if self.mode == 'min':
            self.best = float('inf')
            self.mode_op = lambda current, best: current < best
        else:
            self.best = float('-inf')
            self.mode_op = lambda current, best: current > best
        self.num_bad_epochs = 0

    def on_validation_end(self, trainer, pl_module):
        logs = trainer.logged_metrics
        current = logs.get(self.monitor)
        if current is None:
            return
        if self.mode_op(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        if self.num_bad_epochs > self.patience:
            if self.verbose:
                print(f"Epoch {trainer.current_epoch}: reducing cosine max LR by factor {self.factor} due to no improvement in {self.monitor} for {self.patience + 1} epochs")
            for scheduler_cfg in trainer.lr_scheduler_configs:
                scheduler = scheduler_cfg.scheduler
                if isinstance(scheduler, CosineAnnealingLR):
                    old_bases = scheduler.base_lrs
                    new_bases = [base_lr * self.factor for base_lr in old_bases]
                    scheduler.base_lrs = new_bases
                    scheduler.last_epoch = -1  # Reset the cycle to start from new max
                    # Set current LR to new max for immediate effect
                    optimizer = scheduler.optimizer
                    for param_group, new_base in zip(optimizer.param_groups, new_bases):
                        param_group['lr'] = new_base
            self.num_bad_epochs = 0