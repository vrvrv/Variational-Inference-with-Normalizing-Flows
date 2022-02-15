import os
import torch
import numpy as np
from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader


class GLMMDataset(Dataset):
    def __init__(
            self,
            data_dir: str
    ):
        data = np.load(os.path.join(data_dir, "glmm_data.npz"))
        x = torch.from_numpy(data['x'])
        y = torch.from_numpy(data['y'])
        z = torch.from_numpy(data['z'])

        self.xzy = torch.cat([x, z, y], dim=-1).float()

    def __getitem__(self, idx):
        return self.xzy[idx], torch.Tensor([0])

    def __len__(self):
        return len(self.xzy)


class GLMMDataModule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
            self,
            data_dir: str,
            batch_size: int,
            **kwargs
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        self.data_train = GLMMDataset(self.data_dir)
        self.total_sample = len(self.data_train)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.total_sample,
            shuffle=False,
        )