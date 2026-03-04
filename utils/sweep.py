from datetime import datetime
from dataclasses import replace
from typing import Callable

import wandb

import config as conf
from utils.tools import setup_logger

logger = setup_logger("sweep")

_ABBREV = {
    "use_pos_enc": "pos", "use_sigmoid": "sig", "num_layers": "L",
    "loss_type": "loss", "scenario": "sc", "num_heads": "H", "use_phase_bias": "pb",
}


def _cfg_from_wandb(wc: dict) -> conf.ExperimentConfig:
    model_overrides = {k.split("model.", 1)[1]: v for k, v in wc.items() if k.startswith("model.")}
    top_overrides   = {k: v for k, v in wc.items() if not k.startswith("model.")}
    model_cfg = replace(conf.SWEEP_BASE.model, **model_overrides)
    return replace(conf.SWEEP_BASE, model=model_cfg, file_path="", predict_file_path="", **top_overrides)


def _run_name(wc: dict) -> str:
    base = conf.SWEEP_BASE
    diffs = {
        k.split(".")[-1]: v for k, v in wc.items()
        if v != getattr(getattr(base, "model", base), k.split(".")[-1], getattr(base, k, None))
    }
    parts = [f"{_ABBREV.get(k, k)}={int(v) if isinstance(v, bool) else v}" for k, v in diffs.items()]
    return "_".join(parts) or "baseline"


def run_sweep_agent(
    run_experiment: Callable,
    sweep_config: dict = conf.WANDB_SWEEP_CONFIG,
) -> None:
    named_config = {**sweep_config, "name": datetime.now().strftime("%Y%m%d_%H%M")}
    sweep_id = wandb.sweep(named_config, project=conf.SWEEP_BASE.name)
    logger.info(f"Sweep created: {sweep_id}")

    def _agent_fn():
        run = wandb.init()
        wc = dict(run.config)
        run.name = _run_name(wc)
        run_experiment(_cfg_from_wandb(wc), sweep_run_id=run.id)

    wandb.agent(sweep_id, function=_agent_fn)
