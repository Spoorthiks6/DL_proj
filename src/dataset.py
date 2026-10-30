"""Omniglot loading, character-level splits, and random pair generation."""
import random

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .utils import DATA_DIR

# Omniglot images are dark ink (0) on white (255); keep that convention everywhere.
TRAIN_TF = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=255),
    transforms.ToTensor(),
])
EVAL_TF = transforms.ToTensor()


class OmniglotSet(Dataset):
    """Thin wrapper over torchvision Omniglot that indexes images by character."""

    def __init__(self, background: bool, transform, root=DATA_DIR):
        self.ds = datasets.Omniglot(str(root), background=background, download=True, transform=transform)
        self.labels = [c for _, c in self.ds._flat_character_images]
        self.by_class = {}
        for i, c in enumerate(self.labels):
            self.by_class.setdefault(c, []).append(i)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i]  # (image tensor, character id)


def split_characters(base: OmniglotSet, val_chars: int = 100, seed: int = 42):
    """Split the 964 background characters into train / validation characters.
    Validation characters are never trained on (used for checkpoint choice + threshold)."""
    classes = sorted(base.by_class)
    random.Random(seed).shuffle(classes)
    return classes[val_chars:], classes[:val_chars]


class PairDataset(Dataset):
    """Endless supply of random pairs: 50% same character (y=1), 50% different (y=0)."""

    def __init__(self, base: OmniglotSet, classes, size: int):
        self.base, self.classes, self.size = base, list(classes), size

    def __len__(self):
        return self.size

    def __getitem__(self, _):
        c1 = random.choice(self.classes)
        if random.random() < 0.5:
            i1, i2 = random.sample(self.base.by_class[c1], 2)
            y = 1.0
        else:
            c2 = random.choice(self.classes)
            while c2 == c1:
                c2 = random.choice(self.classes)
            i1 = random.choice(self.base.by_class[c1])
            i2 = random.choice(self.base.by_class[c2])
            y = 0.0
        return self.base[i1][0], self.base[i2][0], torch.tensor(y)
