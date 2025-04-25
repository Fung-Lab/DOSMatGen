import os

import torch

from torch.utils.data import Dataset
from torch_geometric.data import Data

from dosmatgen.utils.data import preprocess, add_scaled_lattice_prop

class CrystalDataset(Dataset):
    def __init__(
        self, 
        name,
        dataset_path,
        targets_path,
        prop,
        niggli,
        primitive,
        graph_method,
        preprocess_workers,
        lattice_scale_method,
        save_path,
        tolerance,
    ) -> None:
        super().__init__()
        self.name = name
        self.dataset_path = dataset_path
        self.targets_path = targets_path
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.preprocess_workers = preprocess_workers
        self.lattice_scale_method = lattice_scale_method
        self.save_path = save_path
        self.tolerance = tolerance

        self.preprocess()
        add_scaled_lattice_prop(self.cached_data, self.lattice_scale_method)

    def preprocess(self):
        # Preprocess the raw data
        if os.path.exists(self.save_path) and os.path.isfile(self.save_path):
            self.cached_data = torch.load(self.save_path, weights_only=False)
            return

        # Load raw data
        cached_data = preprocess(
            self.dataset_path,
            self.targets_path,
            self.preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
            prop_list=[self.prop],
            tolerance=self.tolerance,
        )
        if os.path.isfile(self.save_path):
            torch.save(cached_data, self.save_path)
        self.cached_data = cached_data
    
    def __len__(self):
        return len(self.cached_data)
    
    def __repr__(self) -> str:
        return f"CrystalDataset({self.name}, {self.path})"

    def __getitem__(self, idx):
        data_dict = self.cached_data[idx]

        prop = torch.tensor(data_dict[self.prop])
        prop = prop.squeeze()
        prop_dim = prop.dim()

        # scaler is set in the datamodule
        if prop_dim == 1:
            prop = self.scaler.transform(prop)
            prop = prop.view(1, -1)
        else:
            prop = self.scaler.transform(prop)

        (
            frac_coords, 
            atom_types, 
            lengths, 
            angles, 
            edge_indices, 
            to_jimages, 
            num_atoms
        ) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            structure_id=data_dict['structure_id'],
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            y=prop,
        )

        return data