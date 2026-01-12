"""Data loading and preprocessing for CCPD (Chinese City Parking Dataset)."""

import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import torch
import typer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pytorch_lightning as pl


PROVINCES = [
    "皖",
    "沪",
    "津",
    "渝",
    "冀",
    "晋",
    "蒙",
    "辽",
    "吉",
    "黑",
    "苏",
    "浙",
    "京",
    "闽",
    "赣",
    "鲁",
    "豫",
    "鄂",
    "湘",
    "粤",
    "桂",
    "琼",
    "川",
    "贵",
    "云",
    "藏",
    "陕",
    "甘",
    "青",
    "宁",
    "新",
    "警",
    "学",
    "O",
]
ALPHABETS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "O",
]
ADS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "O",
]

CHARS = PROVINCES + ALPHABETS + [c for c in ADS if c not in ALPHABETS]
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARS)}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}
NUM_CLASSES = len(CHARS)
BLANK_IDX = NUM_CLASSES


def parse_ccpd_filename(filename: str) -> dict[str, Any]:
    """Parse CCPD filename to extract annotations.

    Filename format: area-tilt-bbox-vertices-plate_indices-brightness-blur.jpg
    Example: 025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg

    Args:
        filename: The CCPD image filename (without path).

    Returns:
        Dictionary with parsed annotations.
    """
    stem = Path(filename).stem
    parts = stem.split("-")

    if len(parts) < 7:
        raise ValueError(f"Invalid CCPD filename format: {filename}")

    bbox_str = parts[2]
    bbox_coords = bbox_str.split("_")
    left_up = bbox_coords[0].split("&")
    right_down = bbox_coords[1].split("&")

    x_min = int(left_up[0])
    y_min = int(left_up[1])
    x_max = int(right_down[0])
    y_max = int(right_down[1])

    vertices_str = parts[3]
    vertices_parts = vertices_str.split("_")
    vertices = []
    for vp in vertices_parts:
        coords = vp.split("&")
        vertices.append((int(coords[0]), int(coords[1])))

    plate_indices_str = parts[4]
    plate_indices = [int(i) for i in plate_indices_str.split("_")]

    plate_chars = []
    plate_chars.append(PROVINCES[plate_indices[0]])
    plate_chars.append(ALPHABETS[plate_indices[1]])
    for idx in plate_indices[2:]:
        plate_chars.append(ADS[idx])
    plate_text = "".join(plate_chars)

    return {
        "bbox": [x_min, y_min, x_max, y_max],
        "vertices": vertices,
        "plate_text": plate_text,
        "plate_indices": plate_indices,
        "brightness": int(parts[5]) if len(parts) > 5 else None,
        "blur": int(parts[6]) if len(parts) > 6 else None,
    }


def plate_text_to_indices(plate_text: str) -> list[int]:
    """Convert plate text to character indices for CTC loss.

    Args:
        plate_text: License plate text string.

    Returns:
        List of character indices.
    """
    indices = []
    province_char = plate_text[0]
    if province_char in PROVINCES:
        indices.append(CHAR_TO_IDX[province_char])
    else:
        indices.append(CHAR_TO_IDX["O"])

    for char in plate_text[1:]:
        if char in CHAR_TO_IDX:
            indices.append(CHAR_TO_IDX[char])
        else:
            indices.append(CHAR_TO_IDX["O"])
    return indices


def indices_to_plate_text(indices: list[int]) -> str:
    """Convert character indices back to plate text.

    Args:
        indices: List of character indices.

    Returns:
        License plate text string.
    """
    chars = []
    for idx in indices:
        if idx < len(CHARS):
            char = IDX_TO_CHAR[idx]
            if char != "O":
                chars.append(char)
    return "".join(chars)


class CCPDDetectionDataset(Dataset):
    """Dataset for license plate detection training with YOLO."""

    def __init__(
        self,
        data_dir: Path,
        split_file: Path | None = None,
        transform=None,
        max_images: int | None = None,
    ) -> None:
        """Initialize the detection dataset.

        Args:
            data_dir: Root directory containing CCPD images.
            split_file: Optional file with list of image paths for this split.
            transform: Optional transforms to apply.
            max_images: Maximum number of images to use.
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_paths: list[Path] = []

        if split_file and split_file.exists():
            with open(split_file) as f:
                for line in f:
                    img_path = self.data_dir / line.strip()
                    if img_path.exists():
                        self.image_paths.append(img_path)
        else:
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                self.image_paths.extend(self.data_dir.rglob(ext))

        self.image_paths = sorted(self.image_paths)
        if max_images and len(self.image_paths) > max_images:
            self.image_paths = self.image_paths[:max_images]

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with image and bounding box.
        """
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            annotations = parse_ccpd_filename(img_path.name)
            bbox = annotations["bbox"]
        except (ValueError, IndexError):
            bbox = [0, 0, 1, 1]

        h, w = image.shape[:2]
        x_min, y_min, x_max, y_max = bbox
        x_center = ((x_min + x_max) / 2) / w
        y_center = ((y_min + y_max) / 2) / h
        width = (x_max - x_min) / w
        height = (y_max - y_min) / h

        if self.transform:
            transformed = self.transform(image=image, bboxes=[[x_center, y_center, width, height, 0]])
            image = transformed["image"]
            if transformed["bboxes"]:
                x_center, y_center, width, height, _ = transformed["bboxes"][0]

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        target = torch.tensor([0, x_center, y_center, width, height], dtype=torch.float32)

        return {
            "image": image,
            "target": target,
            "image_path": str(img_path),
        }


class CCPDOCRDataset(Dataset):
    """Dataset for license plate OCR training."""

    def __init__(
        self,
        data_dir: Path,
        split_file: Path | None = None,
        img_height: int = 32,
        img_width: int = 100,
        transform=None,
        max_images: int | None = None,
    ) -> None:
        """Initialize the OCR dataset.

        Args:
            data_dir: Root directory containing CCPD images.
            split_file: Optional file with list of image paths.
            img_height: Target image height for OCR.
            img_width: Target image width for OCR.
            transform: Optional additional transforms.
            max_images: Maximum number of images to use.
        """
        self.data_dir = Path(data_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform
        self.image_paths: list[Path] = []
        self.annotations: list[dict] = []

        if split_file and split_file.exists():
            with open(split_file) as f:
                for line in f:
                    img_path = self.data_dir / line.strip()
                    if img_path.exists():
                        self.image_paths.append(img_path)
        else:
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                self.image_paths.extend(self.data_dir.rglob(ext))

        self.image_paths = sorted(self.image_paths)
        if max_images and len(self.image_paths) > max_images:
            self.image_paths = self.image_paths[:max_images]

        for img_path in self.image_paths:
            try:
                ann = parse_ccpd_filename(img_path.name)
                self.annotations.append(ann)
            except (ValueError, IndexError):
                self.annotations.append(None)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a cropped license plate sample for OCR.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with cropped plate image and label.
        """
        img_path = self.image_paths[idx]
        annotation = self.annotations[idx]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if annotation:
            x_min, y_min, x_max, y_max = annotation["bbox"]
            h, w = image.shape[:2]
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
            plate_img = image[y_min:y_max, x_min:x_max]
            plate_text = annotation["plate_text"]
        else:
            plate_img = image
            plate_text = "O" * 7

        if plate_img.size == 0:
            plate_img = image

        plate_img = cv2.resize(plate_img, (self.img_width, self.img_height))

        if self.transform:
            plate_img = self.transform(image=plate_img)["image"]

        if not isinstance(plate_img, torch.Tensor):
            plate_img = torch.from_numpy(plate_img).permute(2, 0, 1).float() / 255.0

        label_indices = plate_text_to_indices(plate_text)
        label = torch.tensor(label_indices, dtype=torch.long)

        return {
            "image": plate_img,
            "label": label,
            "label_length": torch.tensor(len(label_indices), dtype=torch.long),
            "plate_text": plate_text,
            "image_path": str(img_path),
        }


def ocr_collate_fn(batch: list[dict]) -> dict[str, Any]:
    """Collate function for OCR DataLoader to handle variable length labels.

    Args:
        batch: List of sample dictionaries.

    Returns:
        Batched dictionary with padded labels.
    """
    images = torch.stack([item["image"] for item in batch])
    labels = [item["label"] for item in batch]
    label_lengths = torch.stack([item["label_length"] for item in batch])
    plate_texts = [item["plate_text"] for item in batch]

    max_len = max(len(label) for label in labels)
    padded_labels = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, label in enumerate(labels):
        padded_labels[i, : len(label)] = label

    return {
        "images": images,
        "labels": padded_labels,
        "label_lengths": label_lengths,
        "plate_texts": plate_texts,
    }


class CCPDDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for CCPD dataset."""

    def __init__(
        self,
        data_dir: str | Path,
        split_dir: str | Path | None = None,
        task: str = "detection",
        batch_size: int = 32,
        num_workers: int = 4,
        img_height: int = 32,
        img_width: int = 100,
        max_train_images: int = 5000,
        max_val_images: int = 1000,
    ) -> None:
        """Initialize the DataModule.

        Args:
            data_dir: Root directory containing CCPD images.
            split_dir: Directory containing train/val/test split files.
            task: Either 'detection' or 'ocr'.
            batch_size: Batch size for dataloaders.
            num_workers: Number of dataloader workers.
            img_height: Image height for OCR task.
            img_width: Image width for OCR task.
            max_train_images: Maximum number of training images.
            max_val_images: Maximum number of validation images.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split_dir = Path(split_dir) if split_dir else None
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_height = img_height
        self.img_width = img_width
        self.max_train_images = max_train_images
        self.max_val_images = max_val_images

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        """Setup datasets for each stage.

        Supports two data layouts:
        1. Flat: all images in data_dir
        2. Split: data_dir/train/ and data_dir/val/ subfolders

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        dataset_class = CCPDDetectionDataset if self.task == "detection" else CCPDOCRDataset
        kwargs = {}
        if self.task == "ocr":
            kwargs = {"img_height": self.img_height, "img_width": self.img_width}

        train_dir = self.data_dir / "train" if (self.data_dir / "train").exists() else self.data_dir
        val_dir = self.data_dir / "val" if (self.data_dir / "val").exists() else self.data_dir

        if train_dir != self.data_dir:
            print(f"Using pre-split dataset: train={train_dir}, val={val_dir}")

        if stage == "fit" or stage is None:
            train_split = self.split_dir / "train.txt" if self.split_dir else None
            val_split = self.split_dir / "val.txt" if self.split_dir else None

            self.train_dataset = dataset_class(
                train_dir, split_file=train_split, max_images=self.max_train_images, **kwargs
            )
            self.val_dataset = dataset_class(val_dir, split_file=val_split, max_images=self.max_val_images, **kwargs)

        if stage == "test" or stage is None:
            test_dir = self.data_dir / "test" if (self.data_dir / "test").exists() else val_dir
            test_split = self.split_dir / "test.txt" if self.split_dir else None
            self.test_dataset = dataset_class(test_dir, split_file=test_split, max_images=self.max_val_images, **kwargs)

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        collate = ocr_collate_fn if self.task == "ocr" else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        collate = ocr_collate_fn if self.task == "ocr" else None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        collate = ocr_collate_fn if self.task == "ocr" else None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate,
            pin_memory=True,
        )


def export_yolo_format(
    data_dir: Path,
    output_dir: Path,
    split_file: Path | None = None,
    max_images: int = 50000,
) -> None:
    """Export CCPD dataset to YOLO format for training with Ultralytics.

    Args:
        data_dir: Root directory containing CCPD images.
        output_dir: Output directory for YOLO format dataset.
        split_file: Optional split file.
        max_images: Maximum number of images to export (default: 50000).
    """
    start_time = time.time()

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning for images in {data_dir}...")
    image_paths = []
    if split_file and split_file.exists():
        print(f"Using split file: {split_file}")
        with open(split_file) as f:
            for line in f:
                img_path = data_dir / line.strip()
                if img_path.exists():
                    image_paths.append(img_path)
    else:
        print("No split file provided, scanning all images...")
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_paths.extend(data_dir.rglob(ext))

    if len(image_paths) > max_images:
        print(f"Limiting to {max_images} images (found {len(image_paths)})")
        image_paths = image_paths[:max_images]

    print(f"Processing {len(image_paths)} images")

    exported = 0
    skipped_parse = 0
    skipped_read = 0

    for img_path in tqdm(image_paths, desc="Exporting to YOLO format", unit="img"):
        try:
            annotation = parse_ccpd_filename(img_path.name)
        except (ValueError, IndexError):
            skipped_parse += 1
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            skipped_read += 1
            continue
        h, w = image.shape[:2]

        x_min, y_min, x_max, y_max = annotation["bbox"]
        x_center = ((x_min + x_max) / 2) / w
        y_center = ((y_min + y_max) / 2) / h
        width = (x_max - x_min) / w
        height = (y_max - y_min) / h

        shutil.copy(img_path, images_dir / img_path.name)

        label_file = labels_dir / (img_path.stem + ".txt")
        with open(label_file, "w") as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        exported += 1

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print("Export complete!")
    print(f"  Exported: {exported} images")
    print(f"  Skipped (parse error): {skipped_parse}")
    print(f"  Skipped (read error): {skipped_read}")
    print(f"  Output directory: {output_dir}")
    print(f"  Time elapsed: {elapsed:.1f}s ({exported/elapsed:.1f} img/s)")
    print(f"{'='*50}")


def preprocess(data_path: Path, output_folder: Path) -> None:
    """Preprocess CCPD data and export to YOLO format.

    Args:
        data_path: Path to raw CCPD data.
        output_folder: Output directory.
    """
    print("Preprocessing data...")
    export_yolo_format(data_path, output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
