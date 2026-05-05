import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def split_by_engine(groups, val_ratio: float = 0.2, seed: int = 42):
    unique_engines = np.unique(groups)

    np.random.seed(seed)
    np.random.shuffle(unique_engines)

    split_idx = int(len(unique_engines) * (1 - val_ratio))

    train_engines = unique_engines[:split_idx]
    val_engines = unique_engines[split_idx:]

    train_mask = np.isin(groups, train_engines)
    val_mask = np.isin(groups, val_engines)

    return train_mask, val_mask


class RULDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_loaders(X_train, y_train, X_val, y_val, batch_size: int = 128):
    train_loader = DataLoader(
        RULDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        RULDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader
