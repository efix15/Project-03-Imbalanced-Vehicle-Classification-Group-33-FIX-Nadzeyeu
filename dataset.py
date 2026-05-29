"""
Dataset loader for the Aerial Vehicle Classification dataset.
Expected folder structure:
    data/
        train/
            0/   (Sedan images)
            1/   (SUV images)
            ...
            9/   (Flatbed truck w/ trailer)
        val/
            0/
            ...
            9/
"""

import os
from PIL import Image
from torch.utils.data import Dataset


class VehicleDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        """
        Args:
            root    : root data directory containing 'train/' and 'val/' folders
            split   : 'train' or 'val'
            transform: torchvision transforms to apply
        """
        self.root = os.path.join(root, split)
        self.transform = transform
        self.samples = []   # list of (image_path, label) tuples

        for class_idx in sorted(os.listdir(self.root)):
            class_dir = os.path.join(self.root, class_idx)
            if not os.path.isdir(class_dir):
                continue
            label = int(class_idx)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
