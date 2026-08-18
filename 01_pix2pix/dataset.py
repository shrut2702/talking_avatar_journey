"""Facade dataset: pairs a segmentation mask (.png) with its photo (.jpg).

Each __getitem__ returns (mask, img) so the mask is the network input x and the
photo is the target y.
"""

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def default_transform(size=256):
    """The transform used in the notebook: resize to size x size, scale to [0, 1]."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])


class FacadeDataset(Dataset):
    def __init__(self, path, transforms=None, train: bool = True, split: float = 0.8, random_seed: int = 42):
        folder = Path(path)
        files = {}
        for file in folder.rglob("*"):
            if file.is_file() and file.suffix.lower() in {".jpg", ".png"}:
                files.setdefault(file.stem, {})[file.suffix.lower()] = str(file)

        img_paths = [
            (files[name][".jpg"], files[name][".png"])
            for name in files
            if ".jpg" in files[name] and ".png" in files[name]
        ]

        random.seed(random_seed)
        random.shuffle(img_paths)

        if train:
            self.img_paths = img_paths[:int(len(img_paths) * split)]
        else:
            self.img_paths = img_paths[int(len(img_paths) * split):]

        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_paths[index][0]
        mask_path = self.img_paths[index][1]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if self.transforms:
            img = self.transforms(img)
            mask = self.transforms(mask)

        return mask, img

    def __len__(self):
        return len(self.img_paths)
