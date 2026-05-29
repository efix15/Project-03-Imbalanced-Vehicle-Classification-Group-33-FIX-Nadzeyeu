"""
Imbalanced Vehicle Classification - Training Script
EARIN Project - Summer 2026

Usage examples
--------------
# Structure A – flat folder (your case: EO/ with subfolders 0-9)
python train.py --data_dir path/to/EO

# Structure B – pre-split folders (data/train/ + data/val/)
python train.py --data_dir data

# Change backbone, epochs, etc.
python train.py --data_dir path/to/EO --backbone efficientnet --epochs 30
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import VehicleDataset

# ── Class names ───────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Sedan", "SUV", "Pickup truck", "Van", "Box truck",
    "Motorcycle", "Flatbed truck", "Bus",
    "Pickup truck w/ trailer", "Flatbed truck w/ trailer"
]

# ── Expected training samples per class (for loss weighting) ─────────────────
TRAIN_COUNTS = [234209, 20089, 15301, 10655, 1741, 852, 828, 624, 840, 633]


# ── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ── Imbalance handling ────────────────────────────────────────────────────────

def get_weighted_sampler(dataset):
    """WeightedRandomSampler — balances batches across minority classes."""
    # Count actual samples per class from the loaded dataset
    label_counts = {}
    for _, label in dataset.samples:
        label_counts[label] = label_counts.get(label, 0) + 1

    # Fall back to TRAIN_COUNTS if a class has 0 samples (edge case)
    num_classes = len(CLASS_NAMES)
    counts = np.array([label_counts.get(i, 1) for i in range(num_classes)],
                      dtype=float)
    class_weights = 1.0 / counts
    sample_weights = [class_weights[label] for _, label in dataset.samples]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def get_loss_weights(device, dataset=None):
    """Inverse-frequency weights for CrossEntropyLoss."""
    if dataset is not None:
        # Compute from actual dataset so it works for any split ratio
        label_counts = {}
        for _, label in dataset.samples:
            label_counts[label] = label_counts.get(label, 0) + 1
        counts = torch.tensor(
            [float(label_counts.get(i, 1)) for i in range(len(CLASS_NAMES))]
        )
    else:
        counts = torch.tensor(TRAIN_COUNTS, dtype=torch.float)

    weights = 1.0 / counts
    weights = weights / weights.sum() * len(CLASS_NAMES)   # normalize
    return weights.to(device)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes=10, backbone="resnet50"):
    if backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model


# ── Train / eval loops ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return running_loss / total, correct / total, macro_f1, all_preds, all_labels


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, save_path="confusion_matrix.png"):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_curves(train_losses, val_losses, train_accs, val_accs,
                val_f1s, save_path="training_curves.png"):
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, train_losses, label="Train")
    axes[0].plot(epochs, val_losses, label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_accs, label="Train")
    axes[1].plot(epochs, val_accs, label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, val_f1s, color="green", label="Val Macro F1")
    axes[2].set_title("Macro F1-Score (Validation)")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Data dir    : {args.data_dir}")
    print(f"Split ratio : {args.split_ratio} train / {1-args.split_ratio:.2f} val")

    # Datasets — auto-detects flat vs pre-split structure
    train_dataset = VehicleDataset(
        root=args.data_dir, split="train",
        transform=get_transforms(train=True),
        split_ratio=args.split_ratio, seed=args.seed
    )
    val_dataset = VehicleDataset(
        root=args.data_dir, split="val",
        transform=get_transforms(train=False),
        split_ratio=args.split_ratio, seed=args.seed
    )
    # SOLUTION INTELLIGENTE : On prend un échantillon de CHAQUE classe pour éviter les plantages
    by_class_train = {}
    for path, label in train_dataset.samples:
        by_class_train.setdefault(label, []).append((path, label))
    train_dataset.samples = []
    for label, items in by_class_train.items():
        train_dataset.samples.extend(items[:100])  # 100 images max par classe

    by_class_val = {}
    for path, label in val_dataset.samples:
        by_class_val.setdefault(label, []).append((path, label))
    val_dataset.samples = []
    for label, items in by_class_val.items():
        val_dataset.samples.extend(items[:20])   # 20 images max par classe

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val   samples: {len(val_dataset)}")

    # Sampler + loaders
    sampler = get_weighted_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=sampler, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # Model
    model = build_model(num_classes=10, backbone=args.backbone).to(device)

    # Loss with class weights derived from actual training split
    criterion = nn.CrossEntropyLoss(
        weight=get_loss_weights(device, dataset=train_dataset)
    )

    # Optimizer + cosine scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_f1, best_epoch = 0.0, 0
    history = {k: [] for k in ("train_loss", "val_loss",
                                "train_acc", "val_acc", "val_f1")}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1, preds, labels = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        for key, val in zip(history.keys(),
                            [train_loss, val_loss, train_acc, val_acc, val_f1]):
            history[key].append(val)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), args.save_path)
            print(f"  → Best model saved (F1={best_f1:.4f})")

    print(f"\nTraining complete. Best Macro F1: {best_f1:.4f} at epoch {best_epoch}")

    # Final evaluation with best model
    model.load_state_dict(torch.load(args.save_path))
    _, _, _, preds, labels = evaluate(model, val_loader, criterion, device)

    print("\n── Classification Report ──")
    print(classification_report(labels, preds, target_names=CLASS_NAMES))

    os.makedirs("outputs", exist_ok=True)
    plot_confusion_matrix(labels, preds, "outputs/confusion_matrix.png")
    plot_curves(
        history["train_loss"], history["val_loss"],
        history["train_acc"], history["val_acc"],
        history["val_f1"], "outputs/training_curves.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle Classification Training")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Flat root (EO/) OR parent of train/val/")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "efficientnet"])
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--split_ratio", type=float, default=0.8,
                        help="Train fraction when auto-splitting (default 0.8)")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed for reproducible split")
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--save_path",   type=str,   default="best_model.pth")
    args = parser.parse_args()
    main(args)
