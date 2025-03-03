# Data Loading and Preprocessing
import torch
from torchvision.transforms import Compose, Normalize, Lambda
from torch.utils.data import DataLoader
from medmnist.dataset import OrganMNIST3D


def to_tensor_3d(x):
    if x.ndim == 3:
        x = x[None, ...]
    return torch.from_numpy(x).float()


transform = Compose([
    Lambda(to_tensor_3d),
    Normalize(mean=[0.5], std=0.5)
])


def get_dataloader(batch_size=16, root="/root/.medmnist"):
    train_dataset = OrganMNIST3D(
        split="train", transform=transform, download=True, root=root)
    val_dataset = OrganMNIST3D(
        split="val", transform=transform, download=True, root=root)
    test_dataset = OrganMNIST3D(
        split="test", transform=transform, download=True, root=root)
    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    }
