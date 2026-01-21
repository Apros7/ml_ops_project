"""Models for license plate detection and OCR."""

from pathlib import Path
from shutil import copyfile

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from ultralytics import YOLO
from loguru import logger

from ml_ops.data import (
    NUM_CLASSES,
    BLANK_IDX,
    indices_to_plate_text,
    ENGLISH_NUM_CLASSES,
    ENGLISH_BLANK_IDX,
    english_indices_to_plate_text,
)
from ml_ops.profile import torch_profiler, should_profile


class PlateDetector(pl.LightningModule):
    """License plate detector using YOLOv8.

    This wraps Ultralytics YOLO for use with PyTorch Lightning.
    For actual training, it's recommended to use the Ultralytics CLI directly,
    but this wrapper enables integration with Lightning's logging and callbacks.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        num_classes: int = 1,
        learning_rate: float = 1e-3,
        pretrained: bool = True,
    ) -> None:
        """Initialize the plate detector.

        Args:
            model_name: YOLOv8 model variant (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x).
            num_classes: Number of classes (1 for license plate detection).
            learning_rate: Learning rate for training.
            pretrained: Whether to use pretrained weights.
        """
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.model = YOLO(model_name)

    def forward(self, x: torch.Tensor) -> list:
        """Run inference.

        Args:
            x: Input images tensor or list of image paths.

        Returns:
            YOLO detection results.
        """
        return self.model(x)

    def predict(self, image_path: str, conf: float = 0.25) -> list[dict]:
        """Run prediction on a single image.

        Args:
            image_path: Path to the image.
            conf: Confidence threshold.

        Returns:
            List of detected plates with bounding boxes.
        """
        results = self.model(image_path, conf=conf)
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    "bbox": box.xyxy[0].tolist(),
                    "confidence": box.conf[0].item(),
                    "class": int(box.cls[0].item()),
                }
                detections.append(detection)

        return detections

    def train_yolo(
        self,
        data_yaml: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        device: str | int = "auto",
        project: str = "runs/detect",
        name: str = "plate_detection",
    ) -> None:
        """Train the YOLO model using Ultralytics native training.

        This is the recommended way to train YOLO models.

        Args:
            data_yaml: Path to YOLO data.yaml config file.
            epochs: Number of training epochs.
            imgsz: Input image size.
            batch: Batch size.
            device: Device to train on ('cpu', 0, 'auto', etc.).
            project: Project directory for saving results.
            name: Experiment name.
        """
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
        )

    def configure_optimizers(self):
        """Configure optimizers (for Lightning integration)."""
        return AdamW(self.parameters(), lr=self.learning_rate)


class CRNN(nn.Module):
    """CRNN (Convolutional Recurrent Neural Network) for OCR.

    Architecture designed specifically for license plate OCR:
    - VGG-style CNN that preserves width (critical for CTC loss)
    - Uses (2,1) pooling to reduce height while keeping width
    - Produces sequence length ~24 for input width 100
    - Bidirectional LSTM for sequence modeling
    - Linear layer for character classification

    Supports two modes:
    - Full mode (default): Predicts all 7 characters including Chinese province
    - English-only mode: Predicts only 6 characters (positions 2-7, no Chinese)
    """

    def __init__(
        self,
        img_height: int = 32,
        img_width: int = 200,
        num_classes: int = NUM_CLASSES,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        english_only: bool = False,
    ) -> None:
        """Initialize CRNN.

        Args:
            img_height: Input image height (must be 32 for architecture).
            img_width: Input image width (200 gives seq_len ~49).
            num_classes: Number of character classes (ignored if english_only=True).
            hidden_size: LSTM hidden size.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate.
            english_only: If True, use English-only character set (34 chars vs 69).
        """
        super().__init__()
        self.english_only = english_only

        # Override num_classes for English-only mode
        if english_only:
            self.num_classes = ENGLISH_NUM_CLASSES  # 34 (A-Z no I/O + 0-9)
        else:
            self.num_classes = num_classes

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Use mode-dependent class count so English-only outputs 34+1 instead of default 69+1
        self.fc = nn.Linear(hidden_size * 2, self.num_classes + 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images of shape (batch, 3, height, width).

        Returns:
            Log probabilities of shape (seq_len, batch, num_classes+1).
        """
        conv_out = self.cnn(x)
        batch, channels, height, width = conv_out.size()
        conv_out = conv_out.squeeze(2)
        conv_out = conv_out.permute(0, 2, 1)

        rnn_out, _ = self.rnn(conv_out)
        output = self.fc(rnn_out)
        output = output.permute(1, 0, 2)
        output = F.log_softmax(output, dim=2)

        return output


class PlateOCR(pl.LightningModule):
    """PyTorch Lightning module for license plate OCR.

    Supports two modes:
    - Full mode: Predicts all 7 characters including Chinese province
    - English-only mode: Predicts only 6 characters (positions 2-7, no Chinese)
      Uses a smaller character set (34 vs 69) which is easier to learn.
    """

    def __init__(
        self,
        img_height: int = 32,
        img_width: int = 200,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        max_epochs: int = 15,
        output_dir: str | None = None,
        english_only: bool = False,
        enable_profiling: bool = False,
        profile_every_n_steps: int = 100,
    ) -> None:
        """Initialize PlateOCR module.

        Args:
            img_height: Input image height.
            img_width: Input image width.
            hidden_size: LSTM hidden size.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate.
            learning_rate: Learning rate.
            max_epochs: Maximum epochs for OneCycleLR scheduler.
            output_dir: Directory to save prediction visualizations.
            english_only: If True, predict only English chars (positions 2-7).
            enable_profiling: Whether to enable profiling.
            profile_every_n_steps: Profile every N training steps.
        """
        super().__init__()
        self.save_hyperparameters()

        self.english_only = english_only
        self.enable_profiling = enable_profiling
        self.profile_every_n_steps = profile_every_n_steps

        self.model = CRNN(
            img_height=img_height,
            img_width=img_width,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            english_only=english_only,
        )

        # Use appropriate blank index based on mode
        blank_idx = ENGLISH_BLANK_IDX if english_only else BLANK_IDX
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.output_dir = Path(output_dir) if output_dir else None

        self.val_samples: list[dict] = []

        if english_only:
            logger.info("English-only mode: Predicting 6 characters (positions 2-7)")
            logger.info(f"   Character set: A-Z (no I,O) + 0-9 = {ENGLISH_NUM_CLASSES} classes")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images.

        Returns:
            Log probabilities for CTC decoding.
        """
        return self.model(x)

    def _calculate_char_accuracy(self, pred: str, gt: str) -> float:
        """Calculate character-level accuracy between prediction and ground truth."""
        if len(gt) == 0:
            return 1.0 if len(pred) == 0 else 0.0
        correct = sum(1 for p, g in zip(pred, gt) if p == g)
        return correct / max(len(pred), len(gt))

    def _calculate_edit_distance(self, pred: str, gt: str) -> int:
        """Calculate Levenshtein edit distance between prediction and ground truth."""
        if len(pred) == 0:
            return len(gt)
        if len(gt) == 0:
            return len(pred)

        # Dynamic programming for edit distance
        dp = [[0] * (len(gt) + 1) for _ in range(len(pred) + 1)]
        for i in range(len(pred) + 1):
            dp[i][0] = i
        for j in range(len(gt) + 1):
            dp[0][j] = j

        for i in range(1, len(pred) + 1):
            for j in range(1, len(gt) + 1):
                if pred[i - 1] == gt[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[len(pred)][len(gt)]

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Batch dictionary with images and labels.
            batch_idx: Batch index.

        Returns:
            Training loss.
        """
        images = batch["images"]
        labels = batch["labels"]
        label_lengths = batch["label_lengths"]
        plate_texts = batch["plate_texts"]

        # Profile training step if enabled
        profile_this_step = should_profile(batch_idx, self.profile_every_n_steps) if self.enable_profiling else False
        output_dir = self.output_dir.parent / "profiling" if self.output_dir else None

        with torch_profiler(
            enabled=profile_this_step,
            output_dir=output_dir,
            trace_name=f"train_step_{batch_idx}",
        ):
            log_probs = self(images)

            seq_len = log_probs.size(0)
            batch_size = log_probs.size(1)
            input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=self.device)

            labels_flat = labels.flatten()

            loss = self.ctc_loss(log_probs, labels_flat, input_lengths, label_lengths)

        # Decode predictions for training metrics (every 50 steps to save compute)
        if batch_idx % 50 == 0:
            preds = self.decode(log_probs)
            train_correct = sum(1 for pred, gt in zip(preds, plate_texts) if pred == gt)
            train_accuracy = train_correct / len(plate_texts)
            train_char_acc = sum(self._calculate_char_accuracy(p, g) for p, g in zip(preds, plate_texts)) / len(preds)
            self.log("train_accuracy", train_accuracy, on_step=True, prog_bar=False)
            self.log("train_char_accuracy", train_char_acc, on_step=True, prog_bar=False)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """Validation step.

        Args:
            batch: Batch dictionary with images and labels.
            batch_idx: Batch index.

        Returns:
            Dictionary with loss and accuracy metrics.
        """
        images = batch["images"]
        labels = batch["labels"]
        label_lengths = batch["label_lengths"]
        plate_texts = batch["plate_texts"]

        # Profile validation step if enabled (only first batch of epoch)
        profile_this_step = self.enable_profiling and batch_idx == 0
        output_dir = self.output_dir.parent / "profiling" if self.output_dir else None

        with torch_profiler(
            enabled=profile_this_step,
            output_dir=output_dir,
            trace_name=f"val_step_epoch_{self.current_epoch}",
        ):
            log_probs = self(images)

            seq_len = log_probs.size(0)
            batch_size = log_probs.size(1)
            input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=self.device)

            labels_flat = labels.flatten()

            loss = self.ctc_loss(log_probs, labels_flat, input_lengths, label_lengths)

        preds = self.decode(log_probs)
        correct = sum(1 for pred, gt in zip(preds, plate_texts) if pred == gt)
        accuracy = correct / len(plate_texts)

        # Calculate additional metrics
        char_accuracies = [self._calculate_char_accuracy(p, g) for p, g in zip(preds, plate_texts)]
        avg_char_accuracy = sum(char_accuracies) / len(char_accuracies)

        edit_distances = [self._calculate_edit_distance(p, g) for p, g in zip(preds, plate_texts)]
        avg_edit_distance = sum(edit_distances) / len(edit_distances)

        # Calculate per-position accuracy (for 6 positions in English-only mode)
        expected_len = 6 if self.english_only else 7
        position_correct = [0] * expected_len
        position_total = [0] * expected_len
        for pred, gt in zip(preds, plate_texts):
            for i in range(min(len(pred), len(gt), expected_len)):
                position_total[i] += 1
                if pred[i] == gt[i]:
                    position_correct[i] += 1

        # Log main metrics
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy, on_epoch=True, prog_bar=True)
        self.log("val_char_accuracy", avg_char_accuracy, on_epoch=True, prog_bar=False)
        self.log("val_edit_distance", avg_edit_distance, on_epoch=True, prog_bar=False)

        # Log per-position accuracy
        for i in range(expected_len):
            if position_total[i] > 0:
                pos_acc = position_correct[i] / position_total[i]
                self.log(f"val_pos_{i+1}_accuracy", pos_acc, on_epoch=True, prog_bar=False)

        if batch_idx == 0 and len(self.val_samples) < 8:
            num_samples = min(8 - len(self.val_samples), images.size(0))
            for i in range(num_samples):
                self.val_samples.append(
                    {
                        "image": images[i].cpu(),
                        "pred": preds[i],
                        "gt": plate_texts[i],
                    }
                )

        return {"val_loss": loss, "val_accuracy": accuracy}

    def on_validation_epoch_end(self) -> None:
        """Save visualization of predictions at the end of each validation epoch."""
        if not self.val_samples or self.output_dir is None:
            self.val_samples = []
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        num_samples = min(8, len(self.val_samples))
        fig, axes = plt.subplots(2, 4, figsize=(16, 6))
        axes = axes.flatten()

        for i, sample in enumerate(self.val_samples[:num_samples]):
            img = sample["image"].permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            axes[i].imshow(img)
            pred, gt = sample["pred"], sample["gt"]
            color = "green" if pred == gt else "red"
            axes[i].set_title(f"Pred: {pred}\nGT: {gt}", fontsize=10, color=color)
            axes[i].axis("off")

        for i in range(num_samples, 8):
            axes[i].axis("off")

        epoch = self.current_epoch
        plt.suptitle(f"Epoch {epoch} - Validation Predictions", fontsize=14)
        plt.tight_layout()

        save_path = self.output_dir / f"epoch_{epoch:03d}_predictions.png"
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"\nSaved prediction visualization to: {save_path}")

        self.val_samples = []

    def decode(self, log_probs: torch.Tensor) -> list[str]:
        """Decode CTC output using greedy decoding.

        Args:
            log_probs: Log probabilities of shape (seq_len, batch, num_classes+1).

        Returns:
            List of decoded plate texts.
        """
        _, max_indices = torch.max(log_probs, dim=2)
        max_indices = max_indices.permute(1, 0)

        # Use appropriate constants based on mode
        if self.english_only:
            blank_idx = ENGLISH_BLANK_IDX
            num_classes = ENGLISH_NUM_CLASSES
            decode_fn = english_indices_to_plate_text
        else:
            blank_idx = BLANK_IDX
            num_classes = NUM_CLASSES
            decode_fn = indices_to_plate_text

        decoded = []
        for indices in max_indices:
            chars = []
            prev_idx = -1
            for idx in indices.tolist():
                if idx != blank_idx and idx != prev_idx:
                    if idx < num_classes:
                        chars.append(idx)
                prev_idx = idx
            decoded.append(decode_fn(chars))

        return decoded

    def predict_step(self, batch: dict, batch_idx: int) -> list[str]:
        """Prediction step.

        Args:
            batch: Batch dictionary with images.
            batch_idx: Batch index.

        Returns:
            List of predicted plate texts.
        """
        images = batch["images"]
        log_probs = self(images)
        return self.decode(log_probs)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        if self.trainer and self.trainer.estimated_stepping_batches:
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,
                anneal_strategy="cos",
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        return optimizer


class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM with a linear projection.

    This matches the common CRNN-style OCR implementation where each LSTM block is:
    LSTM(bidirectional) -> Linear(hidden*2 -> out).
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """Initialize the bidirectional LSTM block.

        Args:
            input_size: Input feature size per timestep.
            hidden_size: Hidden size for the LSTM (per direction).
            output_size: Output feature size per timestep after the linear projection.
        """
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, time, input_size).

        Returns:
            Output tensor of shape (batch, time, output_size).
        """
        recurrent, _ = self.rnn(x)
        return self.linear(recurrent)


class VGGFeatureExtractor(nn.Module):
    """VGG-style feature extractor matching the pretrained OCR `.pth` weights."""

    def __init__(self, input_channel: int = 1, output_channel: int = 256) -> None:
        """Initialize the feature extractor.

        Args:
            input_channel: Number of input image channels.
            output_channel: Number of output feature channels.
        """
        super().__init__()
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, output_channel // 8, 3, 1, 1),  # 0: 1 -> 32 (for output_channel=256)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(output_channel // 8, output_channel // 4, 3, 1, 1),  # 3: 32 -> 64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(output_channel // 4, output_channel // 2, 3, 1, 1),  # 6: 64 -> 128
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channel // 2, output_channel // 2, 3, 1, 1),  # 8: 128 -> 128
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(output_channel // 2, output_channel, 3, 1, 1, bias=False),  # 11: 128 -> 256
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channel, output_channel, 3, 1, 1, bias=False),  # 14: 256 -> 256
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(output_channel, output_channel, 2, 1, 0),  # 18: 256 -> 256
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from an input image tensor."""
        return self.ConvNet(x)


class PretrainedPlateOCR(nn.Module):
    """OCR model loader for the packaged `models/ocr_best.pth` weights (CTC-based)."""

    def __init__(
        self,
        characters: str,
        img_height: int = 32,
        img_width: int = 200,
        input_channel: int = 1,
        output_channel: int = 256,
        hidden_size: int = 256,
    ) -> None:
        """Initialize a CRNN-style OCR model.

        Args:
            characters: String of supported characters in index order (excluding blank).
            img_height: Expected input height.
            img_width: Expected input width.
            input_channel: Number of input channels (1 for grayscale).
            output_channel: Feature extractor output channels.
            hidden_size: LSTM hidden size per direction.
        """
        super().__init__()
        self.characters = characters
        self.img_height = img_height
        self.img_width = img_width
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.hidden_size = hidden_size

        self.FeatureExtraction = VGGFeatureExtractor(input_channel=input_channel, output_channel=output_channel)
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(output_channel, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size),
        )
        self.Prediction = nn.Linear(hidden_size, len(characters) + 1)

    @classmethod
    def from_pth(cls, weights_path: str, map_location: str = "cpu") -> "PretrainedPlateOCR":
        """Load a pretrained OCR model from a `.pth` file.

        Args:
            weights_path: Path to `.pth` weights file.
            map_location: Torch map_location for loading.

        Returns:
            A `PretrainedPlateOCR` instance with loaded weights.
        """
        ckpt = torch.load(weights_path, map_location=map_location)
        if not isinstance(ckpt, dict):
            raise ValueError(f"Unsupported OCR weights format in: {weights_path}")

        characters = ckpt.get("characters", "")
        if not isinstance(characters, str) or not characters:
            raise ValueError("OCR weights missing `characters` string.")

        img_height = int(ckpt.get("img_height", 32))
        img_width = int(ckpt.get("img_width", 200))
        network_params = ckpt.get("network_params", {}) if isinstance(ckpt.get("network_params", {}), dict) else {}

        input_channel = int(network_params.get("input_channel", 1))
        output_channel = int(network_params.get("output_channel", 256))
        hidden_size = int(network_params.get("hidden_size", 256))

        model = cls(
            characters=characters,
            img_height=img_height,
            img_width=img_width,
            input_channel=input_channel,
            output_channel=output_channel,
            hidden_size=hidden_size,
        )

        state_dict = ckpt.get("state_dict", ckpt)
        if not isinstance(state_dict, dict):
            raise ValueError("OCR weights missing `state_dict` mapping.")

        cleaned = {k.removeprefix("module."): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned, strict=True)
        model.eval()
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images of shape (batch, 1, height, width).

        Returns:
            Logits of shape (batch, time, num_classes_with_blank).
        """
        features = self.FeatureExtraction(x)
        features = features.squeeze(2).permute(0, 2, 1)
        contextual = self.SequenceModeling(features)
        return self.Prediction(contextual)

    def decode(self, logits: torch.Tensor) -> list[str]:
        """Decode logits using greedy CTC decoding.

        Args:
            logits: Tensor of shape (batch, time, num_classes_with_blank).

        Returns:
            List of decoded plate strings.
        """
        preds = logits.argmax(dim=2).tolist()
        decoded: list[str] = []
        for seq in preds:
            chars: list[str] = []
            prev = -1
            for idx in seq:
                if idx != 0 and idx != prev:
                    chars.append(self.characters[idx - 1])
                prev = idx
            decoded.append("".join(chars))
        return decoded


class LicensePlateRecognizer:
    """End-to-end license plate recognition pipeline.

    Combines plate detection and OCR for complete license plate recognition.
    """

    def __init__(
        self,
        detector_weights: str = "yolov8n.pt",
        ocr_checkpoint: str | None = None,
        ocr_weights: str | None = None,
        device: str = "auto",
    ) -> None:
        """Initialize the recognizer.

        Args:
            detector_weights: Path to YOLO detector weights.
            ocr_checkpoint: Path to OCR checkpoint (Lightning `.ckpt`) or pretrained `.pth`.
            ocr_weights: Path to pretrained OCR `.pth` weights (preferred).
            device: Device to run inference on.
        """
        self.device = self._get_device(device)

        detector_weights = self._ensure_yolo_weights_suffix(detector_weights)
        self.detector = YOLO(detector_weights)

        ocr_path = ocr_weights or ocr_checkpoint
        if ocr_path and Path(ocr_path).suffix.lower() == ".pth":
            self.ocr = PretrainedPlateOCR.from_pth(ocr_path).to(self.device)
        else:
            self.ocr = PlateOCR()
            if ocr_path:
                self.ocr = PlateOCR.load_from_checkpoint(ocr_path)
            self.ocr = self.ocr.to(self.device)
        self.ocr.eval()

    def _ensure_yolo_weights_suffix(self, weights: str) -> str:
        """Ensure Ultralytics sees a supported `.pt` suffix for weights.

        Some Ultralytics versions enforce `.pt` at prediction-time even if the
        file contents are valid with a `.pth` extension.
        """
        path = Path(weights)
        if path.suffix.lower() != ".pth":
            return weights

        pt_path = path.with_suffix(".pt")
        if pt_path.exists():
            return str(pt_path)

        try:
            copyfile(path, pt_path)
            return str(pt_path)
        except OSError as exc:
            logger.warning(f"Failed to create .pt alias for YOLO weights {path}: {exc}")
            return weights

    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device.

        Args:
            device: Device string ('auto', 'cpu', 'cuda', etc.).

        Returns:
            torch.device object.
        """
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def recognize(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        img_height: int | None = None,
        img_width: int | None = None,
    ) -> list[dict]:
        """Recognize license plates in an image.

        Args:
            image_path: Path to the input image.
            conf_threshold: Detection confidence threshold.
            img_height: Height for OCR input.
            img_width: Width for OCR input.

        Returns:
            List of recognized plates with bounding boxes and text.
        """
        import cv2

        results = self.detector(image_path, conf=conf_threshold)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target_h = img_height or getattr(self.ocr, "img_height", None) or getattr(self.ocr, "hparams", {}).get("img_height", 32)
        target_w = img_width or getattr(self.ocr, "img_width", None) or getattr(self.ocr, "hparams", {}).get("img_width", 200)

        recognitions = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)

                plate_crop = image[y1:y2, x1:x2]

                if plate_crop.size == 0:
                    continue

                plate_resized = cv2.resize(plate_crop, (target_w, target_h))
                if isinstance(self.ocr, PretrainedPlateOCR):
                    plate_gray = cv2.cvtColor(plate_resized, cv2.COLOR_RGB2GRAY)
                    plate_tensor = torch.from_numpy(plate_gray).unsqueeze(0).unsqueeze(0).float() / 255.0
                else:
                    plate_tensor = torch.from_numpy(plate_resized).permute(2, 0, 1).float() / 255.0
                    plate_tensor = plate_tensor.unsqueeze(0)
                plate_tensor = plate_tensor.to(self.device)

                with torch.no_grad():
                    ocr_out = self.ocr(plate_tensor)
                    plate_text = self.ocr.decode(ocr_out)[0]

                recognitions.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": box.conf[0].item(),
                        "plate_text": plate_text,
                    }
                )

        return recognitions


if __name__ == "__main__":
    logger.debug("Testing CRNN model...")
    model = CRNN()
    x = torch.randn(2, 3, 32, 200)
    output = model(x)
    logger.debug(f"Input shape: {x.shape}")
    logger.debug(f"Output shape: {output.shape}")
    logger.debug(f"Expected: (seq_len, batch=2, num_classes+1={NUM_CLASSES + 1})")
