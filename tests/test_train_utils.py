from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from ml_ops import train


def test_normalize_path_none() -> None:
    assert train._normalize_path(None) is None


def test_normalize_path_relative_is_project_rooted() -> None:
    result = train._normalize_path("data/somewhere")
    assert result is not None
    assert result.is_absolute()
    assert str(result).endswith("data/somewhere")


def test_load_hydra_config_is_reentrant() -> None:
    cfg1 = train.load_hydra_config(config_path=train.CONFIGS_DIR, config_name="config")
    cfg2 = train.load_hydra_config(config_path=train.CONFIGS_DIR, config_name="config")
    assert cfg1 is not None
    assert cfg2 is not None
    assert "data" in cfg1
    assert "training" in cfg2


def test_start_wandb_run_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("wandb.init should not be called when disabled")

    monkeypatch.setattr(train.wandb, "init", _boom)
    active = train._start_wandb_run(OmegaConf.create({"enabled": False}), name="x", metadata={"k": "v"})
    assert active is False


def test_start_and_finish_wandb_run_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def _fake_init(*, entity: str, project: str, name: str, config: dict[str, Any]) -> None:
        calls["entity"] = entity
        calls["project"] = project
        calls["name"] = name
        calls["config"] = config

    finished: list[bool] = []

    def _fake_finish() -> None:
        finished.append(True)

    monkeypatch.setattr(train.wandb, "init", _fake_init)
    monkeypatch.setattr(train.wandb, "finish", _fake_finish)

    wandb_cfg = OmegaConf.create({"enabled": True, "entity": "ent", "project": "proj"})
    active = train._start_wandb_run(wandb_cfg, name="run-name", metadata={"a": 1})
    assert active is True
    assert calls == {"entity": "ent", "project": "proj", "name": "run-name", "config": {"a": 1}}

    train._finish_wandb_run(active)
    assert finished == [True]


def test_log_yolo_results_to_wandb_skips_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    logged: list[dict[str, Any]] = []

    def _fake_log(payload: dict[str, Any], step: Any = None) -> None:
        _ = step
        logged.append(payload)

    monkeypatch.setattr(train.wandb, "log", _fake_log)

    missing = tmp_path / "results.csv"
    train._log_yolo_results_to_wandb(missing)
    assert logged == []


def test_log_yolo_results_to_wandb_parses_csv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[dict[str, Any], Any]] = []

    def _fake_log(payload: dict[str, Any], step: Any = None) -> None:
        calls.append((payload, step))

    monkeypatch.setattr(train.wandb, "log", _fake_log)

    results_file = tmp_path / "results.csv"
    results_file.write_text("epoch,metrics/mAP50,mode\n0,0.5,train\n1,0.75,val\n")

    train._log_yolo_results_to_wandb(results_file)

    assert calls[0][0]["epoch"] == 0
    assert calls[0][0]["metrics/mAP50"] == 0.5
    assert calls[0][0]["mode"] == "train"
    assert calls[0][1] == 0


def test_export_best_model_copies_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path / "models")

    src = tmp_path / "src.pt"
    dst = train.MODELS_DIR / "detector" / "best.pt"
    src.write_bytes(b"payload")

    train._export_best_model(src, dst)
    assert dst.exists()
    assert dst.read_bytes() == b"payload"


def test_build_easyocr_character_list_is_unique() -> None:
    chars = train._build_easyocr_character_list(english_only=False)
    assert len(chars) == len(set(chars))
    assert "0" in chars and "9" in chars


def test_normalize_easyocr_plate_text_english_only() -> None:
    assert train._normalize_easyocr_plate_text(" i O l | ", english_only=True) == "1011"


def test_easyocr_collate_fn_outputs_grayscale_and_texts() -> None:
    img1 = torch.zeros(3, 2, 2)
    img1[0] = 1.0  # pure red -> 0.299
    img2 = torch.zeros(3, 2, 2)
    img2[1] = 1.0  # pure green -> 0.587

    batch = [
        {"image": img1, "plate_text": "I O"},
        {"image": img2, "plate_text": "AB"},
    ]

    out = train._easyocr_collate_fn(batch, english_only=True)

    assert out["images"].shape == (2, 1, 2, 2)
    assert out["texts"] == ["10", "AB"]

    expected1 = (0.299 - 0.5) / 0.5
    expected2 = (0.587 - 0.5) / 0.5
    assert torch.allclose(out["images"][0, 0], torch.full((2, 2), expected1))
    assert torch.allclose(out["images"][1, 0], torch.full((2, 2), expected2))


def test_ctc_greedy_decode_removes_blanks_and_repeats() -> None:
    indices = torch.tensor([[0, 1, 1, 0, 2, 2]])
    decoded = train._ctc_greedy_decode(indices, idx_to_char=["", "A", "B"])
    assert decoded == ["AB"]


def test_calculate_char_accuracy_handles_empty_and_mismatch() -> None:
    assert train._calculate_char_accuracy("", "") == 1.0
    assert train._calculate_char_accuracy("A", "") == 0.0
    assert train._calculate_char_accuracy("ABC", "ABCD") == 0.75


def test_get_accelerator_force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    accel, devices = train.get_accelerator(force_cpu=True)
    assert (accel, devices) == ("cpu", "auto")


def test_get_accelerator_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    accel, devices = train.get_accelerator()
    assert (accel, devices) == ("gpu", "auto")


def test_get_accelerator_falls_back_to_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    accel, devices = train.get_accelerator()
    assert (accel, devices) == ("mps", "1")


def test_get_accelerator_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    accel, devices = train.get_accelerator()
    assert (accel, devices) == ("cpu", "auto")


def test_train_detector_requires_split_sizes() -> None:
    cfg = OmegaConf.create(
        {
            "data": {
                "data_dir": "data",
                "split_dir": None,
                "processed_yolo_dir": "data/processed/yolo",
                "train_split": 0.8,
                "max_total_images": None,
                "max_train_images": None,
                "max_val_images": None,
            },
            "training": {"detector": {}, "ocr": {}},
            "model": {"detector": {}, "ocr": {}},
            "wandb_configs": {"enabled": False},
        }
    )

    with pytest.raises(ValueError, match="Cannot determine train/val split sizes"):
        train._train_detector_with_cfg(cfg)


def test_train_detector_writes_data_yaml_and_calls_export(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "train").mkdir(parents=True)
    (data_dir / "val").mkdir(parents=True)

    processed_yolo_dir = tmp_path / "processed" / "yolo"
    project_dir = tmp_path / "runs" / "detect"

    cfg = OmegaConf.create(
        {
            "data": {
                "data_dir": str(data_dir),
                "split_dir": None,
                "processed_yolo_dir": str(processed_yolo_dir),
                "train_split": 0.8,
                "max_total_images": 10,
                "name": "CCPD",
            },
            "training": {"detector": {"project_dir": str(project_dir), "experiment_name": "exp"}, "ocr": {}},
            "model": {"detector": {"model_name": "yolov8n.pt", "num_classes": 1, "pretrained": True}, "ocr": {}},
            "wandb_configs": {"enabled": False},
        }
    )

    export_calls: list[tuple[Path, Path, Path | None, int]] = []

    def _fake_export_yolo_format(
        data_dir_arg: Path, output_dir: Path, split_file: Path | None = None, max_images: int = 50000, **_kwargs: Any
    ) -> None:
        export_calls.append((Path(data_dir_arg), Path(output_dir), split_file, int(max_images)))

    class _FakePlateDetector:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def train_yolo(
            self, *, data_yaml: str, epochs: int, imgsz: int, batch: int, device: Any, project: str, name: str
        ) -> None:
            _ = (epochs, imgsz, batch, device, project, name)
            assert Path(data_yaml).exists()

    monkeypatch.setattr(train, "export_yolo_format", _fake_export_yolo_format)
    monkeypatch.setattr(train, "PlateDetector", _FakePlateDetector)
    monkeypatch.setattr(train, "_start_wandb_run", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(train, "_finish_wandb_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train, "get_accelerator", lambda *_, **__: ("cpu", "auto"))

    _, experiment_name = train._train_detector_with_cfg(cfg, name="exp")
    assert experiment_name == "exp"

    data_yaml = processed_yolo_dir / "data.yaml"
    assert data_yaml.exists()
    assert f"path: {processed_yolo_dir.absolute()}" in data_yaml.read_text()

    assert export_calls[0][0] == data_dir / "train"
    assert export_calls[0][3] == 8
    assert export_calls[1][0] == data_dir / "val"
    assert export_calls[1][3] == 2
