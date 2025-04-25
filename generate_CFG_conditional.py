import argparse

import ase
from ase import Atoms
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from dosmatgen.diffusion.diffusion_cfg import CSPDiffusion
from dosmatgen.utils.constants import cdvae_train_num_elements_distribution
from dosmatgen.utils.utils import decode
from dosmatgen.dataset.datamodule import CrystalDataModule, worker_init_fn
from dosmatgen.dataset.dataset import CrystalDataset

def diffuse(
    data_loader, 
    model,
    n_candidates,
    step_lr, 
    diff_ratio,
):
    all_outputs = {}

    count = 0

    for idx, batch in enumerate(data_loader):
        batch = batch.to('cuda')
        batch_outputs = []
        count += 1
        
        for i in range(n_candidates):
            outputs, _ = model.sample(
                batch,
                step_lr=step_lr,
                diff_ratio=diff_ratio,
                conditional=True
            )

            outputs = {
                'structure_id': batch.structure_id,
                'num_atoms': outputs['num_atoms'].detach().cpu(),
                'atom_types': outputs['atom_types'].detach().cpu(),
                'frac_coords': outputs['frac_coords'].detach().cpu(),
                'lattices': outputs['lattices'].detach().cpu(),
            }
            batch_outputs.append(outputs)
        
        all_outputs[idx] = {
            "batch": batch,
            "outputs": batch_outputs
        }

        # do just one epoch for testing
        # to reconstruct the entire dataset, comment the following line
        break
    
    return all_outputs

def main(args):
    root_path = Path(args.root_path)

    now = datetime.now()
    formatted_time = now.strftime("%d%m%Y_%H%M%S")
    save_path = Path(args.save_path) / formatted_time

    # load config
    print("Loading model...")
    config_path = root_path / 'hparams.yaml'
    config = OmegaConf.load(config_path)

    # load checkpoint
    ckpt_path = glob(str(root_path / '*.ckpt'))
    if len(ckpt_path) == 0:
        raise ValueError("No checkpoint file found.")
    elif len(ckpt_path) > 1:
        raise ValueError("Multiple checkpoint files found.")
    ckpt_path = ckpt_path[0]

    model = CSPDiffusion.load_from_checkpoint(ckpt_path, config=config)
    model.to('cuda')

    # we will use the DOS in the test set as the conditional
    print("Loading test loader....")
    if args.batch_size is not None:
        config.datamodule.batch_size.test = args.batch_size

    data_module = CrystalDataModule(config, scaler_path=str(root_path))
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()

    # diffuse
    print("Denoising for generation...")
    all_outputs = diffuse(
        test_loader,
        model, 
        args.n_candidates,
        args.step_lr,
        args.diff_ratio
    )

    # decode
    print("Decoding to ase.Atoms objects...")
    atoms_list = {}

    for j in tqdm(all_outputs):
        output_dict = all_outputs[j]['outputs'][0]

        start_idx = 0
        for i in range(len(output_dict['structure_id'])):
            sid = output_dict['structure_id'][i]
            num_atoms = output_dict['num_atoms'][i]
            lattices = output_dict['lattices'][i]

            end_idx = start_idx + num_atoms
            atom_types = output_dict['atom_types'][start_idx:end_idx]
            frac_coords = output_dict['frac_coords'][start_idx:end_idx]

            assert len(atom_types) == num_atoms.item() == frac_coords.shape[0]

            atoms = Atoms(
                cell=lattices.cpu().numpy(),
                scaled_positions=frac_coords.cpu().numpy(),
                numbers=atom_types
            )

            atoms_list[sid] = atoms

            start_idx = end_idx

    save_path.mkdir(exist_ok=True)
    for k, v in atoms_list.items():
        ase.io.write(str(save_path / f"{k}.cif"), v)

    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, required=True, help="Path to the folder containing model.ckpt, hparams.yaml, and scaler.pt")
    parser.add_argument('--save_path', type=str, default="structures/", help="Path to save the generated structures")
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--step_lr', type=float, default=5e-6)
    parser.add_argument('--n_candidates', type=float, default=1, help="Number of candidate structures to generate per given sample in the test set")
    parser.add_argument('--diff_ratio', type=float, default=0.5, help="timestep at which denoising starts; (0, 1]")
    args = parser.parse_args()

    main(args)