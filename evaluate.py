"""
Evaluation script - loads a trained model and evaluates on the validation set.
Usage:
    python evaluate.py --data_dir data --model_path best_model.pth --backbone resnet50
"""

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from dataset import VehicleDataset
from train import build_model, CLASS_NAMES


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(labels, preds, save_path):
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
    print(f"Saved: {save_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = VehicleDataset(args.data_dir, split="val",
                             transform=get_val_transforms())
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    model = build_model(num_classes=10, backbone=args.backbone).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Loaded model from {args.model_path}")

    labels, preds, probs = predict(model, loader, device)

    macro_f1 = f1_score(labels, preds, average="macro")
    accuracy = (labels == preds).mean()
    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print("\n── Per-class Report ──")
    print(classification_report(labels, preds, target_names=CLASS_NAMES))

    os.makedirs("outputs", exist_ok=True)
    plot_confusion_matrix(labels, preds, "outputs/confusion_matrix_eval.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_path", type=str, default="best_model.pth")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "efficientnet"])
    args = parser.parse_args()
    main(args)
