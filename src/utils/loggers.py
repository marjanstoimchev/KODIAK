"""Logger setup for M³-Net training."""

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from pathlib import Path
from typing import Optional, Union


def setup_logger(
    logger_type: str,
    experiment_name: str,
    save_dir: str = "logs",
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_tags: Optional[list] = None,
    wandb_notes: Optional[str] = None,
    **kwargs
) -> Union[CSVLogger, TensorBoardLogger, WandbLogger]:
    """
    Setup logger based on type.

    Args:
        logger_type: Type of logger - 'csv', 'wandb', or 'tensorboard'
        experiment_name: Name of the experiment
        save_dir: Directory to save logs
        wandb_project: WandB project name (for wandb logger)
        wandb_entity: WandB entity/username (for wandb logger)
        wandb_tags: Tags for WandB run (for wandb logger)
        wandb_notes: Notes for WandB run (for wandb logger)
        **kwargs: Additional arguments passed to logger

    Returns:
        Configured logger instance
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if logger_type.lower() == "csv":
        logger = CSVLogger(
            save_dir=str(save_dir),
            name=experiment_name,
            **kwargs
        )
        print(f"Using CSV Logger - Logs will be saved to: {save_dir}/{experiment_name}")

    elif logger_type.lower() == "tensorboard":
        logger = TensorBoardLogger(
            save_dir=str(save_dir),
            name=experiment_name,
            **kwargs
        )
        print(f"Using TensorBoard Logger - Run: tensorboard --logdir={save_dir}/{experiment_name}")

    elif logger_type.lower() == "wandb":
        if wandb_project is None:
            raise ValueError("wandb_project must be specified for WandB logger")

        logger = WandbLogger(
            project=wandb_project,
            name=experiment_name,
            save_dir=str(save_dir),
            entity=wandb_entity,
            tags=wandb_tags,
            notes=wandb_notes,
            **kwargs
        )
        print(f"Using WandB Logger - Project: {wandb_project}, Run: {experiment_name}")

    else:
        raise ValueError(
            f"Unknown logger type: {logger_type}. Choose from: csv, tensorboard, wandb"
        )

    return logger


def get_logger_from_config(config) -> Union[CSVLogger, TensorBoardLogger, WandbLogger]:
    """
    Create logger from configuration object.

    Args:
        config: Configuration object with logging settings

    Returns:
        Configured logger instance
    """
    # Get logger type with default to CSV
    logger_type = config.logging.get("logger", "csv")
    experiment_name = config.experiment.name

    # Get save_dir with fallbacks
    save_dir = config.logging.get("save_dir", None)
    if save_dir is None:
        base_dir = config.logging.get("base_dir", "logs")
        save_dir = base_dir

    # Base kwargs
    kwargs = {}

    if logger_type.lower() == "wandb":
        # WandB-specific settings
        try:
            wandb_cfg = config.logging.wandb
            kwargs = {
                "wandb_project": wandb_cfg.project if hasattr(wandb_cfg, 'project') else wandb_cfg.get("project", "kodiak"),
                "wandb_entity": wandb_cfg.get("entity", None),
                "wandb_tags": wandb_cfg.get("tags", None),
                "wandb_notes": wandb_cfg.get("notes", None),
            }
        except AttributeError:
            # No wandb config, use defaults
            kwargs = {"wandb_project": "kodiak"}
    elif logger_type.lower() == "tensorboard":
        # TensorBoard-specific settings
        try:
            tb_name = config.logging.tensorboard.get("name", None)
            if tb_name:
                experiment_name = tb_name
        except AttributeError:
            pass  # No tensorboard config

    return setup_logger(
        logger_type=logger_type,
        experiment_name=experiment_name,
        save_dir=save_dir,
        **kwargs
    )
