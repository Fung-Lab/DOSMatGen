import ase

import torch
from tqdm import tqdm
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from dosmatgen.utils.constants import STATS_KEY

# Adapted from https://github.com/hobogalaxy/lightning-hydra-template/blob/6bf03035107e12568e3e576e82f83da0f91d6a11/src/utils/template_utils.py#L125
def log_hyperparameters(
    cfg: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionally saves:
        - sizes of train, val, test dataset
        - number of trainable model parameters
    Args:
        cfg (DictConfig): [description]
        model (pl.LightningModule): [description]
        trainer (pl.Trainer): [description]
    """
    hparams = OmegaConf.to_container(cfg, resolve=True)

    # save number of model parameters
    hparams[f"{STATS_KEY}/params_total"] = sum(p.numel()
                                               for p in model.parameters())
    hparams[f"{STATS_KEY}/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams[f"{STATS_KEY}/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # (this is just a trick to prevent trainer from logging hparams of model, since we already did that above)
    trainer.logger.log_hyperparams = lambda params: None

def decode(data, save_path):
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(key, value.shape)
    
    num_atoms = data['num_atoms'].squeeze()
    atom_types = data['atom_types'].squeeze()
    frac_coords = data['frac_coords'].squeeze()
    lattices = data['lattices'].squeeze()

    print("num_atoms: ", num_atoms.shape)
    print("atom_types: ", atom_types.shape)
    print("frac_coords: ", frac_coords.shape)
    print("lattices: ", lattices.shape)

    start_idx = 0
    atoms_list = []

    for i, num_atom in enumerate(tqdm(num_atoms.tolist())):
        _frac_coords = frac_coords[start_idx:start_idx+num_atom]
        _atom_types = atom_types[start_idx:start_idx+num_atom]

        # atomic_numbers = (np.argmax(_atom_types, axis=-1) + 1)
        atomic_numbers = _atom_types.tolist()

        cell = lattices[i]

        atoms = ase.Atoms(
            numbers=atomic_numbers,
            scaled_positions=_frac_coords.numpy(),
            cell=cell,
            pbc=[True, True, True]
        )
        
        atoms_list.append(atoms)
        start_idx += num_atom

    for i, each in enumerate(tqdm(atoms_list)):
        ase.io.write(str(save_path / f'{i}.cif'), each)