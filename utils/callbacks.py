from pytorch_lightning.callbacks import TQDMProgressBar
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateFinder
import torch
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


class PeriodicLRFinder(LearningRateFinder):
    """
    Periodically runs LR finder and updates CosineAnnealingWarmRestarts scheduler.

    Args:
        tune_every_n_epochs: Run LR finder every N epochs
        min_lr: Minimum learning rate for LR finder
        max_lr: Maximum learning rate for LR finder
        num_training_steps: Number of steps for LR range test (default: 100)
        mode: 'exponential' or 'linear' for LR range test
        early_stop_threshold: Stop if loss exceeds this multiple of best loss
        update_attr: Whether to update model.lr attribute (default: True)
        min_steps_for_suggestion: Minimum number of data points needed (default: 10)
    """

    def __init__(
            self,
            tune_every_n_epochs=10,
            min_lr=1e-7,
            max_lr=1.0,
            num_training_steps=100,
            mode='exponential',
            early_stop_threshold=4.0,
            update_attr=True,
            min_steps_for_suggestion=10,
            attr_name='lr',  # Explicitly set the attribute name
            **kwargs
    ):
        super().__init__(
            min_lr=min_lr,
            max_lr=max_lr,
            num_training_steps=num_training_steps,
            mode=mode,
            early_stop_threshold=early_stop_threshold,
            update_attr=update_attr,
            attr_name=attr_name,
            **kwargs
        )
        self.tune_every_n_epochs = tune_every_n_epochs
        self.optimal_lr_history = []
        self.min_steps_for_suggestion = min_steps_for_suggestion
        self.last_successful_lr = None

    def on_fit_start(self, *args, **kwargs):
        # Override to prevent automatic LR finding at start
        return

    def on_train_epoch_end(self, trainer, pl_module):
        """Run LR finder at specified intervals"""
        current_epoch = trainer.current_epoch

        # Run LR finder at the end of specified epochs
        # This ensures the dataloader is fresh for the next epoch
        if (current_epoch + 1) % self.tune_every_n_epochs == 0 or current_epoch == 0:
            print(f"\n{'=' * 50}")
            print(f"Running LR Finder after epoch {current_epoch}")
            print(f"Will apply to epoch {current_epoch + 1}")
            print(f"{'=' * 50}")

            try:
                # Run the LR range test
                lr_finder = self.lr_find(trainer, pl_module)

                if lr_finder is not None and hasattr(lr_finder, 'results'):
                    num_points = len(lr_finder.results.get('lr', []))
                    print(f"LR Finder collected {num_points} data points")

                    if num_points >= self.min_steps_for_suggestion:
                        # Get the suggested learning rate
                        suggested_lr = lr_finder.suggestion() * 10

                        if suggested_lr is not None:
                            self.last_successful_lr = suggested_lr
                            self.optimal_lr_history.append({
                                'epoch': current_epoch + 1,  # Will be applied to next epoch
                                'lr': suggested_lr
                            })

                            print(f"Suggested LR: {suggested_lr:.2e}")

                            # Update the CosineAnnealingWarmRestarts scheduler
                            self._update_scheduler(trainer, pl_module, suggested_lr)

                            # Optionally plot the results
                            # fig = lr_finder.plot(suggest=True)
                            # plt.savefig(f'lr_finder_epoch_{current_epoch}.png')
                            # plt.close()
                        else:
                            print(f"Warning: Could not suggest LR at epoch {current_epoch}")
                            if self.last_successful_lr is not None:
                                print(f"Using last successful LR: {self.last_successful_lr:.2e}")
                    else:
                        print(
                            f"Warning: Only {num_points} data points collected (need at least {self.min_steps_for_suggestion})")
                        print("Try increasing num_training_steps or ensure your dataloader has enough batches")
                else:
                    print(f"Warning: LR finder returned no results at epoch {current_epoch}")

            except Exception as e:
                print(f"Error during LR finding at epoch {current_epoch}: {str(e)}")
                print("Skipping LR update for this epoch")
                if self.last_successful_lr is not None:
                    print(f"Continuing with last successful LR: {self.last_successful_lr:.2e}")

    def _update_scheduler(self, trainer, pl_module, new_lr):
        """Update CosineAnnealingWarmRestarts with new base learning rate"""

        # Find the scheduler in the LR scheduler configs
        if hasattr(trainer, 'lr_scheduler_configs') and trainer.lr_scheduler_configs:
            for config in trainer.lr_scheduler_configs:
                scheduler = config.scheduler

                # Check if it's CosineAnnealingWarmRestarts
                if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    print(f"Updating CosineAnnealingWarmRestarts base_lr: {scheduler.base_lrs} -> {new_lr:.2e}")

                    # Get current scheduler parameters
                    T_0 = scheduler.T_0
                    T_mult = scheduler.T_mult
                    eta_min = scheduler.eta_min

                    # Create new scheduler with updated learning rate
                    optimizer = scheduler.optimizer

                    # Update optimizer learning rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr

                    # Create new scheduler instance
                    new_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer,
                        T_0=T_0,
                        T_mult=T_mult,
                        eta_min=eta_min
                    )

                    # Replace the scheduler
                    config.scheduler = new_scheduler

                    print(f"Scheduler updated successfully. T_0={T_0}, T_mult={T_mult}")
                    break

def create_callbacks(scenario: str) -> list[pl.Callback]:
    """Create training callbacks for checkpointing and monitoring."""
    checkpoint_cb = ModelCheckpoint(
        monitor="val/eer",
        mode="min",
        filename=f'{scenario}' + "-{epoch:02d}-{val/eer:.2f}",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False
    )

    lr_monitor = LearningRateMonitor(logging_interval=None)
    validation_silent_bar = ValidationSilentProgressBar()

    return [checkpoint_cb, lr_monitor, validation_silent_bar]

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