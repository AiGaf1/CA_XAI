import os
import logging

from pytorch_metric_learning.losses import SupConLoss, ArcFaceLoss, TripletMarginLoss
from pytorch_lightning.loggers import WandbLogger
import wandb
from datetime import datetime

from models.Litmodel import KeystrokeLitModel
from conf import ExperimentConfig
from models.CNN import CNN_LTE


def setup_logger(name: str = None, level=logging.INFO) -> logging.Logger:
    """Set up and return a logger instance with a consistent format."""
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-2s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
        logger.propagate = False
    return logger

def setup_wandb_logging(config: ExperimentConfig, model_name: str) -> tuple[WandbLogger, str]:
    """Setup W&B logging with appropriate tags and versioning."""
    tags = [
        f"scenario_{config.scenario}",
        # f"embedding_{config.embedding_size}",
        f"window_size_{config.window_size}",
        f"epochs_{config.epochs}",
        f"trigperiods_{config.n_periods}",
        f"head_{config.use_projector}",
        f"model_{model_name}"
    ]

    version = datetime.now().strftime("%Y%m%d_%H%M")
    experiment_name = f'{config.epochs}_{config.scenario}'

    wandb_logger = WandbLogger(
        project=config.name,
        name=experiment_name,
        version=version,
        log_model=True,
        tags=tags
    )

    return wandb_logger, version

def load_comparisons(scenario: str, logger) -> list[tuple[str, str]]:
    """Load comparison pairs from file."""
    comps_file = f"data/{scenario}/{scenario}_comparisons.txt"

    with open(comps_file, "r") as f:
        comparisons = eval(f.readline())

    logger.info(f"Loaded {len(comparisons)} comparison pairs")
    return comparisons

def save_predictions(similarities: list[float], scenario: str, logger) -> None:
    """Save similarity scores to file."""
    output_file = f'{scenario}_predictions.txt'
    with open(output_file, "w") as f:
        f.write(str(similarities))
    logger.info(f"Predictions saved to {output_file}")

def export_to_onnx(config: ExperimentConfig,ckpt_path: str, wandb_logger: WandbLogger, model):
    import torch.serialization
    from pytorch_metric_learning.distances.cosine_similarity import CosineSimilarity

    # Allow the custom class
    torch.serialization.add_safe_globals([CosineSimilarity, SupConLoss, ArcFaceLoss, TripletMarginLoss])

    # After creating your model
    batch_size = config.batches_per_epoch_train
    seq_len = config.window_size
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dummy_x = torch.cat([
        torch.randn(batch_size, seq_len, 1),
        torch.randn(batch_size, seq_len, 1),
        torch.randint(0, 256, (batch_size, seq_len, 1))
    ], dim=-1).to(device)

    dummy_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    dummy_x = dummy_x.to(device)
    dummy_mask = dummy_mask.to(device)

    model = model.to(device)

    # Parse score from checkpoint filename (e.g., "mobile-769-1.49.ckpt" -> 1.49)
    filename = os.path.basename(ckpt_path)
    score_str = filename.split('-')[-1].replace('.ckpt', '')
    float_score = float(score_str)
    rounded_score = round(float_score, 2)

    best_model = KeystrokeLitModel.load_from_checkpoint(
        ckpt_path,
        model=model,
        strict=False
    )
    best_model.eval()

    # Construct ONNX save path in the parent directory under 'onnx' subfolder
    ckpt_dir = os.path.dirname(ckpt_path)  # e.g., "Keystroke-XAI/20251227_0330/checkpoints"
    parent_dir = os.path.dirname(ckpt_dir)  # e.g., "Keystroke-XAI/20251227_0330"
    onnx_dir = os.path.join(parent_dir, 'onnx')
    os.makedirs(onnx_dir, exist_ok=True)  # Create 'onnx' folder if it doesn't exist
    onnx_path = os.path.join(onnx_dir, f"model_{rounded_score}.onnx")

    with torch.no_grad():
        torch.onnx.export(
            best_model,
            (dummy_x, dummy_mask),
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input', 'mask'],
            output_names=['embedding'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'mask': {0: 'batch_size'},
                'embedding': {0: 'batch_size'}
            },
            dynamo = True
        )

    # Extract run_id from a path (assuming structure like "Keystroke-XAI/20251227_0330/checkpoints/...")
    path_parts = ckpt_path.split(os.sep)
    run_id = path_parts[-3]

    artifact = wandb.Artifact(
        name=f"model-onnx-{run_id}-{rounded_score}",
        type="model",
        description="ONNX exported model"
    )

    artifact.add_file(local_path=onnx_path)

    wandb_logger.experiment.log_artifact(artifact)
    artifact.wait()