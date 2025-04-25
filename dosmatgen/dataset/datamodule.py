from typing import Optional, Sequence

import random
import numpy as np
from pathlib import Path 

import torch

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from dosmatgen.dataset.dataset import CrystalDataset
from dosmatgen.utils.data import get_scaler_from_data_list

def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)

class CrystalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: DictConfig,
        scaler_path=None,
    ):
        super().__init__()
        self.config = config

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        self.get_scaler(scaler_path)

    def prepare_data(self) -> None:
        # download only
        pass

    def get_scaler(self, scaler_path):
        # Load once to compute property scaler
        if scaler_path is None:
            train_dataset = self.get_dataset('train')

            self.lattice_scaler = get_scaler_from_data_list(
                train_dataset.cached_data,
                key='scaled_lattice',
            )
            self.scaler = get_scaler_from_data_list(
                train_dataset.cached_data,
                key=train_dataset.prop,
                pred_node_level=self.config.diffusion.model.pred_node_level
            )
        else:
            try:
                self.lattice_scaler = torch.load(
                    Path(scaler_path) / 'lattice_scaler.pt'
                )
                self.scaler = torch.load(
                    Path(scaler_path) / 'prop_scaler.pt'
                )
            except:
                train_dataset = self.get_dataset('train')

                self.lattice_scaler = get_scaler_from_data_list(
                    train_dataset.cached_data,
                    key='scaled_lattice',
                )
                self.scaler = get_scaler_from_data_list(
                    train_dataset.cached_data,
                    key=train_dataset.prop,
                    pred_node_level=self.config.diffusion.model.pred_node_level
                )

    def get_dataset(self, split="train"):
        return CrystalDataset(
            **self.config.datamodule.datasets[split],
            prop=self.config.property,
            niggli=self.config.niggli,
            primitive=self.config.primitive,
            graph_method=self.config.graph_method,
            preprocess_workers=self.config.preprocess_workers,
            lattice_scale_method=self.config.lattice_scale_method,
            tolerance=self.config.tolerance,
        )

    def setup(self, stage: Optional[str]=None):
        """
        construct datasets and assign data scalers.
        """
        if stage is None or stage == "fit":
            self.train_dataset = self.get_dataset('train')
            self.val_dataset = self.get_dataset('val')

            self.train_dataset.lattice_scaler = self.lattice_scaler
            self.train_dataset.scaler = self.scaler

            self.val_dataset.lattice_scaler = self.lattice_scaler
            self.val_dataset.scaler = self.scaler

        if stage is None or stage == "test":
            self.test_dataset = self.get_dataset('test')
            self.test_dataset.lattice_scaler = self.lattice_scaler
            self.test_dataset.scaler = self.scaler

    def setup_adhoc_dataset(self, dataset_path):
        return CrystalDataset(
            name='adhoc',
            dataset_path=dataset_path,
            targets_path=None,
            save_path=dataset_path,
            prop=self.config.property,
            niggli=self.config.niggli,
            primitive=self.config.primitive,
            graph_method=self.config.graph_method,
            preprocess_workers=self.config.preprocess_workers,
            lattice_scale_method=self.config.lattice_scale_method,
            tolerance=self.config.tolerance,
        )

    def get_adhoc_dataloader(self, dataset_path, batch_size=1):
        adhoc_dataset = self.setup_adhoc_dataset(dataset_path)
        adhoc_dataset.lattice_scaler = self.lattice_scaler
        adhoc_dataset.scaler = self.scaler

        return DataLoader(
            adhoc_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=1,
            worker_init_fn=worker_init_fn,
        )

    def train_dataloader(self, shuffle=True) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=shuffle,
            batch_size=self.config.datamodule.batch_size.train,
            num_workers=1,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.config.datamodule.batch_size.val,
            num_workers=1,
            worker_init_fn=worker_init_fn,
        )

    def test_dataloader(self) -> Sequence[DataLoader]:
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.config.datamodule.batch_size.test,
            num_workers=1,
            worker_init_fn=worker_init_fn,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
        )