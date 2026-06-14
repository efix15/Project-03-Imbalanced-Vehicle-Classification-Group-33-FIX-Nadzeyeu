"""
Imbalanced Vehicle Classification - Training Script
EARIN Project - Summer 2026

Phase 3 changes:
  - No resize — images are already 32x32 in the dataset
  - No augmentation — keeping original images as-is
  - Full dataset used — no subsampling
  - Class weights and sampler computed dynamically from real dataset counts
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


def get_transforms(train=True):
    # No resize — images are already 32x32 in the dataset
    # No augmentation — keeping original images as-is
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def get_class_counts(dataset):
    """Compute real class counts from the loaded dataset (no hardcoding)."""
    counts = np.zeros(len(CLASS_NAMES), dtype=float)
    for _, label in dataset.samples:
        counts[label] += 1
    return counts


def get_weighted_sampler(dataset):
    """WeightedRandomSampler — probability of picking an image is inversely
    proportional to its class frequency. Computed from real dataset counts."""
    counts = get_class_counts(dataset)
    print("Train samples per class:")
    for i, (name, c) in enumerate(zip(CLASS_NAMES, counts)):
        print(f"  Class {i} ({name}): {int(c)}")
    class_weights = 1.0 / np.where(counts > 0, counts, 1)
    sample_weights = [class_weights[label] for _, label in dataset.samples]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def get_loss_weights(dataset, device):
    """Inverse-frequency weights for CrossEntropyLoss. Computed from real
    dataset counts so it adapts to any training split automatically."""
    counts = torch.tensor(get_class_counts(dataset), dtype=torch.float)
    weights = 1.0 / torch.clamp(counts, min=1.0)
    weights = weights / weights.sum() * len(CLASS_NAMES)
    return weights.to(device)


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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets
    train_dataset = VehicleDataset(
        root=args.data_dir, split="train", transform=get_transforms(train=True)
    )
    val_dataset = VehicleDataset(
        root=args.data_dir, split="val", transform=get_transforms(train=False)
    )

    # Sampler + loaders — full dataset, no subsampling
    sampler = get_weighted_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=sampler, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"\nTrain size: {len(train_dataset):,} images")
    print(f"Val size  : {len(val_dataset):,} images\n")

    # Model
    model = build_model(num_classes=10, backbone=args.backbone).to(device)

    # Loss weights computed dynamically from real dataset counts
    criterion = nn.CrossEntropyLoss(weight=get_loss_weights(train_dataset, device))

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_f1, best_epoch = 0.0, 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1, preds, labels = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), args.save_path)
            print(f"  -> Best model saved (F1={best_f1:.4f})")

    print(f"\nTraining complete. Best Macro F1: {best_f1:.4f} at epoch {best_epoch}")

    # Final evaluation with best model
    model.load_state_dict(torch.load(args.save_path))
    _, _, _, preds, labels = evaluate(model, val_loader, criterion, device)

    print("\n-- Classification Report --")
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
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "efficientnet"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_path", type=str, default="best_model.pth")
    args = parser.parse_args()
    main(args)
