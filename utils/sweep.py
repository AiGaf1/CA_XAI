"""
Grid sweep runner — edit SWEEP_GRID and SWEEP_BASE in config.py, then run:
    python utils/sweep.py
"""
import copy
import itertools
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ExperimentConfig, SWEEP_GRID, SWEEP_BASE, SWEEP_EXCLUDE
from train import run_experiment

DRY_RUN = False  # set True to print configs without running
INDEX   = None   # set to an int to run only that combination (0-based)


def _build_configs(grid: dict, base: ExperimentConfig) -> list[ExperimentConfig]:
    top_keys    = {k: v for k, v in grid.items() if not k.startswith("model.")}
    nested_keys = {k[len("model."):]: v for k, v in grid.items() if k.startswith("model.")}

    top_combos    = list(itertools.product(*top_keys.values()))
    nested_combos = list(itertools.product(*nested_keys.values())) if nested_keys else [()]

    configs = []
    for top_vals in top_combos:
        top_overrides = dict(zip(top_keys.keys(), top_vals))
        # Clear derived path fields so __post_init__ recomputes them from the new scenario.
        if "scenario" in top_overrides:
            top_overrides.setdefault("file_path", "")
            top_overrides.setdefault("predict_file_path", "")

        for nested_vals in nested_combos:
            nested_overrides = dict(zip(nested_keys.keys(), nested_vals))
            new_model = replace(base.model, **nested_overrides) if nested_overrides else copy.copy(base.model)
            cfg = replace(base, **top_overrides, model=new_model)
            if not any(all(getattr(cfg, k) == v for k, v in excl.items()) for excl in SWEEP_EXCLUDE):
                configs.append(cfg)

    return configs


def _describe(cfg: ExperimentConfig, idx: int, total: int) -> str:
    model_fields = {k: getattr(cfg.model, k.split(".")[-1]) for k in SWEEP_GRID if k.startswith("model.")}
    top_fields   = {k: getattr(cfg, k) for k in SWEEP_GRID if not k.startswith("model.")}
    parts = [f"[{idx+1}/{total}]"] + [f"{k}={v}" for k, v in {**top_fields, **model_fields}.items()]
    return "  ".join(parts)


def main():
    configs = _build_configs(SWEEP_GRID, SWEEP_BASE)
    total   = len(configs)

    print(f"Grid sweep: {total} combinations")
    for i, cfg in enumerate(configs):
        print(f"  {_describe(cfg, i, total)}")

    if DRY_RUN:
        return

    to_run = [configs[INDEX]] if INDEX is not None else configs
    for i, cfg in enumerate(to_run):
        actual_i = INDEX if INDEX is not None else i
        print(f"\n{'='*60}\nRunning {_describe(cfg, actual_i, total)}\n{'='*60}")
        run_experiment(cfg)


if __name__ == "__main__":
    main()
