import argparse
import numpy as np
from pathlib import Path
from glob import glob
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from dosmatgen.diffusion.diffusion_cfg import CSPDiffusion
from dosmatgen.utils.constants import cdvae_train_num_elements_distribution
from dosmatgen.utils.utils import decode

class SampleDataset(Dataset):
    def __init__(self, dataset, total_num):
        super().__init__()
        self.total_num = total_num
        self.distribution = cdvae_train_num_elements_distribution[dataset]

        # sample number of atoms from the training dataset distribution
        self.num_atoms = np.random.choice(
            len(self.distribution), 
            total_num, 
            p=self.distribution
        )
        self.is_carbon = dataset == 'carbon_24'

    def __len__(self) -> int:
        return self.total_num

    def __getitem__(self, index):
        num_atom = self.num_atoms[index]
        data = Data(
            num_atoms=torch.LongTensor([num_atom]),
            num_nodes=num_atom,
            y=torch.ones(num_atom, 400) * 10
        )
        if self.is_carbon:
            data.atom_types = torch.LongTensor([6] * num_atom)
        return data

def diffuse(model, loader, step_lr):
    num_atoms = []
    atom_types = []
    frac_coords = []
    lattices = []

    for idx, batch in enumerate(loader):
        print(f"Denoising batch {idx}...")
        batch = batch.to('cuda')

        # unconditional = True for unconditional generation (without DOS input)
        outputs, traj = model.sample(batch, step_lr=step_lr, unconditional=True)

        num_atoms.append(outputs['num_atoms'].detach().cpu())
        atom_types.append(outputs['atom_types'].detach().cpu())
        frac_coords.append(outputs['frac_coords'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())

    num_atoms = torch.cat(num_atoms, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    frac_coords = torch.cat(frac_coords, dim=0)
    lattices = torch.cat(lattices, dim=0)

    return num_atoms, atom_types, frac_coords, lattices

def main(args):
    root_path = Path(args.root_path)

    now = datetime.now()
    formatted_time = now.strftime("%d%m%Y_%H%M%S")
    save_path = Path(args.save_path) / formatted_time
    save_path.mkdir(exist_ok=True)

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

    test_set = SampleDataset('mp_20', args.batch_size * args.num_batches)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    step_lr = args.step_lr

    # diffuse
    print("Denoising for generation...")
    (
        num_atoms, 
        atom_types, 
        frac_coords, 
        lattices
    ) = diffuse(model, test_loader, step_lr)

    data = {
        'eval_setting': args,
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'frac_coords': frac_coords,
        'lattices': lattices
    }

    torch.save(data, save_path / 'unconditional.pt')

    # decode
    print("Decoding to ase.Atoms objects...")
    decode(data, save_path)

    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, required=True, help="Path to the folder containing model.ckpt, hparams.yaml, and scaler.pt")
    parser.add_argument('--save_path', type=str, default="structures/", help="Path to save the generated structures")
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_batches', type=int, default=1)
    parser.add_argument('--step_lr', type=float, default=5e-6)
    args = parser.parse_args()

    main(args)