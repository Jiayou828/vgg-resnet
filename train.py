"""Train and compare plain and residual VGG-16 on CIFAR-10."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import VGG16CIFAR, count_parameters


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
SUMMARY_FIELDS = (
    "run_name",
    "model",
    "batch_norm",
    "seed",
    "best_epoch",
    "best_val_accuracy",
    "test_loss",
    "test_accuracy",
    "parameters",
    "device",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--model", choices=("plain", "residual"), help="Override model in config.")
    parser.add_argument("--batch-norm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--run-name", help="Optional unique name for this run's output folder.")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve()
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")

    for key in ("model", "batch_norm", "epochs", "batch_size", "num_workers", "seed", "device"):
        value = getattr(args, key.replace("-", "_"), None)
        if value is not None:
            config[key] = value

    if config.get("model") not in ("plain", "residual"):
        raise ValueError("config.model must be 'plain' or 'residual'")
    if config.get("epochs", 0) < 1 or config.get("batch_size", 0) < 1:
        raise ValueError("epochs and batch_size must be positive")
    if config.get("num_workers", -1) < 0:
        raise ValueError("num_workers cannot be negative")
    if config.get("device") not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("device must be auto, cpu, cuda, or mps")

    project_dir = config_path.parent
    for key in ("data_dir", "output_dir"):
        path = Path(config[key]).expanduser()
        config[key] = (project_dir / path).resolve() if not path.is_absolute() else path.resolve()
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def stratified_split(labels: list[int], seed: int) -> tuple[list[int], list[int]]:
    """Return fixed 45,000/5,000 indices, with 500 validation items per class."""
    if len(labels) != 50_000:
        raise ValueError(f"Expected CIFAR-10's 50,000 training labels, got {len(labels)}")
    classes = sorted(set(labels))
    if len(classes) != 10 or any(labels.count(label) != 5_000 for label in classes):
        raise ValueError("Expected ten balanced CIFAR-10 classes with 5,000 items each")

    generator = torch.Generator().manual_seed(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for label in classes:
        class_indices = torch.tensor([index for index, item in enumerate(labels) if item == label])
        order = class_indices[torch.randperm(len(class_indices), generator=generator)].tolist()
        val_indices.extend(order[:500])
        train_indices.extend(order[500:])
    return train_indices, val_indices


def fixed_split_indices(labels: list[int], seed: int, output_root: Path) -> tuple[list[int], list[int]]:
    """Persist the exact split so future runs reuse these same sample indices."""
    split_path = output_root / "splits" / f"seed_{seed}.json"
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        if split.get("seed") != seed:
            raise ValueError(f"Seed mismatch in saved split: {split_path}")
        train_indices = [int(index) for index in split["train_indices"]]
        val_indices = [int(index) for index in split["val_indices"]]
        all_indices = train_indices + val_indices
        if (
            len(train_indices) != 45_000
            or len(val_indices) != 5_000
            or len(set(all_indices)) != 50_000
            or min(all_indices) != 0
            or max(all_indices) != 49_999
            or any(sum(labels[index] == label for index in val_indices) != 500 for label in range(10))
        ):
            raise ValueError(f"Saved split is invalid for the current CIFAR-10 data: {split_path}")
        return train_indices, val_indices

    train_indices, val_indices = stratified_split(labels, seed)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        json.dumps(
            {"seed": seed, "train_indices": train_indices, "val_indices": val_indices},
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return train_indices, val_indices


def make_loaders(config: dict[str, Any], device: torch.device) -> tuple[DataLoader, DataLoader]:
    root = str(config["data_dir"])
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    train_transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
    )
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])

    index_dataset = datasets.CIFAR10(root=root, train=True, download=True)
    train_indices, val_indices = fixed_split_indices(
        index_dataset.targets, config["seed"], config["output_dir"]
    )
    train_dataset = Subset(
        datasets.CIFAR10(root=root, train=True, download=False, transform=train_transform), train_indices
    )
    val_dataset = Subset(
        datasets.CIFAR10(root=root, train=True, download=False, transform=eval_transform), val_indices
    )

    generator = torch.Generator().manual_seed(config["seed"])
    common = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": config["num_workers"] > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader


def make_test_loader(config: dict[str, Any], device: torch.device) -> DataLoader:
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    dataset = datasets.CIFAR10(
        root=str(config["data_dir"]),
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        persistent_workers=config["num_workers"] > 0,
    )


def choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if name == "mps" and not mps_available:
        raise RuntimeError("MPS was requested but is not available")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if mps_available else "cpu")
    return torch.device(name)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size
    return total_loss / total, correct / total


def write_epoch_log(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("epoch", "learning_rate", "train_loss", "train_accuracy", "val_loss", "val_accuracy"))
        writer.writeheader()
        writer.writerows(rows)


def save_curves(path: Path, rows: list[dict[str, Any]]) -> None:
    epochs = [row["epoch"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in rows], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in rows], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch")
    axes[1].plot(epochs, [100 * row["train_accuracy"] for row in rows], label="Train")
    axes[1].plot(epochs, [100 * row["val_accuracy"] for row in rows], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="%")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def append_summary(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    config = load_config(args)
    seed_everything(config["seed"])
    device = choose_device(config["device"])
    model_name = config["model"]
    bn_tag = "bn" if config["batch_norm"] else "no-bn"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{model_name}_{bn_tag}_seed{config['seed']}_{timestamp}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
        raise ValueError("run-name may contain only letters, numbers, dot, underscore, and hyphen")

    output_root: Path = config["output_dir"]
    run_dir = output_root / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run output already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    serializable_config = dict(config)
    serializable_config["data_dir"] = str(config["data_dir"])
    serializable_config["output_dir"] = str(output_root)
    (run_dir / "config.json").write_text(json.dumps(serializable_config, indent=2), encoding="utf-8")

    train_loader, val_loader = make_loaders(config, device)
    model = VGG16CIFAR(residual=model_name == "residual", batch_norm=config["batch_norm"]).to(device)
    parameters = count_parameters(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["min_learning_rate"])
    rows: list[dict[str, Any]] = []
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = 0
    best_checkpoint = run_dir / "best.pt"

    print(f"Run: {run_name} | device: {device} | parameters: {parameters:,}")
    for epoch in range(1, config["epochs"] + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        rows.append(row)
        write_epoch_log(run_dir / "epochs.csv", rows)
        print(
            f"Epoch {epoch:03d}/{config['epochs']} "
            f"train loss {train_loss:.4f} acc {train_accuracy:.4f} | "
            f"val loss {val_loss:.4f} acc {val_accuracy:.4f}"
        )
        if val_accuracy > best_accuracy or (val_accuracy == best_accuracy and val_loss < best_loss):
            best_accuracy, best_loss, best_epoch = val_accuracy, val_loss, epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model": model_name,
                    "batch_norm": config["batch_norm"],
                    "seed": config["seed"],
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "parameters": parameters,
                },
                best_checkpoint,
            )

    save_curves(run_dir / "curves.png", rows)
    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy = run_epoch(model, make_test_loader(config, device), criterion, device)
    result = {
        "run_name": run_name,
        "model": model_name,
        "batch_norm": config["batch_norm"],
        "seed": config["seed"],
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "parameters": parameters,
        "device": str(device),
    }
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    append_summary(output_root / "summary.csv", result)
    print(f"Best validation accuracy: {best_accuracy:.4f} at epoch {best_epoch}")
    print(f"Held-out test accuracy: {test_accuracy:.4f}")
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
