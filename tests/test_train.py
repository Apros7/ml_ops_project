import pytest
from omegaconf import OmegaConf

from ml_ops.train import _train_ocr_with_cfg


def test_train_ocr_requires_max_images() -> None:
    cfg = OmegaConf.create(
        {
            "data": {
                "data_dir": "data/ccpd_tiny",
                "split_dir": None,
                "train_split": 0.8,
                "max_total_images": None,
                "num_workers": 0,
            },
            "training": {
                "ocr": {
                    "batch_size": 2,
                    "max_epochs": 1,
                    "learning_rate": 1e-3,
                    "project_dir": "runs/ocr",
                    "experiment_name": "test_ocr",
                    "patience": 1,
                    "precision": 32,
                    "gradient_clip_val": 5.0,
                }
            },
            "model": {
                "ocr": {
                    "english_only": True,
                    "img_height": 32,
                    "img_width": 64,
                    "hidden_size": 16,
                    "num_layers": 1,
                    "dropout": 0.0,
                }
            },
            "wandb_configs": {"enabled": False},
        }
    )

    with pytest.raises(ValueError, match="Missing value for max_images"):
        _train_ocr_with_cfg(cfg)
