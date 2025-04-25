import re
import json
import ase.io
import numpy as np
import pandas as pd
from p_tqdm import p_umap
from pathlib import Path
from glob import glob

from pymatgen.analysis import local_env
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.graphs import StructureGraph

import torch

from dosmatgen.utils.constants import EPSILON


CrystalNN = local_env.CrystalNN(
    distance_cutoffs=None, 
    x_diff_weight=-1, 
    porous_adjustment=False
)

def preprocess(
    dataset_path,
    targets_path,
    num_workers,
    niggli=False,
    primitive=False,
    graph_method="standard",
    prop_list=None,
    tolerance=0.01,
):
    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)
    elif isinstance(dataset_path, list):
        results = p_umap(
            process_one_cif,
            dataset_path,
            [niggli] * len(dataset_path),
            [primitive] * len(dataset_path),
            [graph_method] * len(dataset_path),
            [prop_list] * len(dataset_path),
            [tolerance] * len(dataset_path),
            num_cpus=num_workers,
        )

        results = sorted(results, key=lambda d: alphanumeric_key(d["structure_id"]))
        return results

    
    if dataset_path.suffix == ".json":
        print(f"Loading JSON file of size {dataset_path.stat().st_size / 1024**2:.2f} MB ...")
        f = open(dataset_path, "r")
        data = json.load(f)
        f.close()

        results = p_umap(
            process_one_json,
            data,
            [niggli] * len(data),
            [primitive] * len(data),
            [graph_method] * len(data),
            [prop_list] * len(data),
            [tolerance] * len(data),
            num_cpus=num_workers,
        )
    elif dataset_path.suffix == ".csv":
        print(f"Loading CSV file of size {dataset_path.stat().st_size / 1024**2:.2f} MB ...")
        df = pd.read_csv(dataset_path)
        results = p_umap(
            process_one_csv,
            [df.iloc[idx] for idx in range(len(df))],
            [niggli] * len(df),
            [primitive] * len(df),
            [graph_method] * len(df),
            [prop_list] * len(df),
            [tolerance] * len(df),
            num_cpus=num_workers,
        )
    elif dataset_path.is_dir():
        print(f"Loading CIF files from directory {dataset_path} ...")
        cif_files = glob(str(dataset_path / "*.cif"))

        results = p_umap(
            process_one_cif,
            cif_files,
            [niggli] * len(cif_files),
            [primitive] * len(cif_files),
            [graph_method] * len(cif_files),
            [prop_list] * len(cif_files),
            [tolerance] * len(cif_files),
            num_cpus=num_workers,
        )

        results = sorted(results, key=lambda d: alphanumeric_key(d["structure_id"]))

    return results

def alphanumeric_key(f):
    base = f.split("/")[-1].split(".")[0]
    parts = re.split(r'(\d+)', base)
    return [int(part) if part.isdigit() else part for part in parts]

def process_one_csv(
    row,
    niggli,
    primitive,
    graph_method,
    prop_list,
    tolerance
):
    """
    Process a single row of a CSV file containing crystal structures.
    """
    crystal_str = row['cif']
    
    # get properties
    props = {
       k: row[k] for k in prop_list if k in row.keys()
    }

    # get canonical crystal
    structure = Structure.from_str(crystal_str, fmt="cif")

    if primitive:
        structure = structure.get_primitive_structure()
    
    if niggli:
        structure = structure.get_reduced_structure()
    
    canonical_structure = Structure(
        lattice=Lattice.from_parameters(*structure.lattice.parameters),
        species=structure.species,
        coords=structure.frac_coords,
        coords_are_cartesian=False
    )

    # output
    graph_arrays = build_crystal_graph(canonical_structure, graph_method)

    return {
        "structure_id": row['material_id'],
        "spacegroup": 1,
        "graph_arrays": graph_arrays,
        **props
    }

def process_one_json(
    data_dict,
    niggli,
    primitive,
    graph_method,
    prop_list,
    tolerance
):
    """
    Process a single JSON file containing crystal structures.
    """
    structure_id = data_dict["structure_id"]
    atomic_numbers = data_dict["atomic_numbers"]
    positions = data_dict["positions"]
    cell = data_dict["cell"]
    
    # get properties
    props = {
        k: data_dict[k] for k in prop_list if k in data_dict
    }

    # get canonical crystal
    structure = Structure(
        cell,
        atomic_numbers,
        positions,
        coords_are_cartesian=True
    )

    if primitive:
        structure = structure.get_primitive_structure()
    
    if niggli:
        structure = structure.get_reduced_structure()
    
    canonical_structure = Structure(
        lattice=Lattice.from_parameters(*structure.lattice.parameters),
        species=structure.species,
        coords=structure.frac_coords,
        coords_are_cartesian=False
    )

    # output
    graph_arrays = build_crystal_graph(canonical_structure, graph_method)

    return {
        "structure_id": structure_id,
        "spacegroup": 1,
        "graph_arrays": graph_arrays,
        **props
    }

def process_one_cif(
    cif,
    niggli,
    primitive,
    graph_method,
    prop_list,
    tolerance
):
    cif_path = Path(cif)
    structure_id = cif_path.stem

    atoms = ase.io.read(cif_path)
    atomic_numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()
    cell = atoms.get_cell()

    props = {
        k: 0 for k in prop_list
    }

    # get canonical crystal
    structure = Structure(
        cell,
        atomic_numbers,
        positions,
        coords_are_cartesian=True
    )

    if primitive:
        structure = structure.get_primitive_structure()
    
    if niggli:
        structure = structure.get_reduced_structure()

    canonical_structure = Structure(
        lattice=Lattice.from_parameters(*structure.lattice.parameters),
        species=structure.species,
        coords=structure.frac_coords,
        coords_are_cartesian=False
    )

    # output
    graph_arrays = build_crystal_graph(canonical_structure, graph_method)

    return {
        "structure_id": structure_id,
        "spacegroup": 1,
        "graph_arrays": graph_arrays,
        **props
    }

def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15
    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)

def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """
    Converts lattice from abc, angles to matrix.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])

def build_crystal_graph(crystal, graph_method='crystalnn'):
    """
    """
    if graph_method == 'crystalnn':
        try:
            crystal_graph = StructureGraph.from_local_env_strategy(
                crystal, 
                CrystalNN
            )
        except:
            crystalNN_tmp = local_env.CrystalNN(
                distance_cutoffs=None, 
                x_diff_weight=-1, 
                porous_adjustment=False, 
                search_cutoff=10
            )
            crystal_graph = StructureGraph.from_local_env_strategy(
                crystal, crystalNN_tmp
            ) 
    elif graph_method == 'none':
        pass
    else:
        raise NotImplementedError

    frac_coords = crystal.frac_coords
    atom_types = crystal.atomic_numbers
    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    assert np.allclose(
        crystal.lattice.matrix,
        lattice_params_to_matrix(*lengths, *angles)
    )

    edge_indices, to_jimages = [], []
    if graph_method != 'none':
        for i, j, to_jimage in crystal_graph.graph.edges(data='to_jimage'):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)
    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)
    num_atoms = atom_types.shape[0]

    return (
        frac_coords, 
        atom_types, 
        lengths, 
        angles, 
        edge_indices, 
        to_jimages, 
        num_atoms
    )

def add_scaled_lattice_prop(data_list, lattice_scale_method):
    for dict in data_list:
        graph_arrays = dict['graph_arrays']
        # the indices are brittle if more objects are returned
        lengths = graph_arrays[2]
        angles = graph_arrays[3]
        num_atoms = graph_arrays[-1]
        assert lengths.shape[0] == angles.shape[0] == 3
        assert isinstance(num_atoms, int)

        if lattice_scale_method == 'scale_length':
            lengths = lengths / float(num_atoms)**(1/3)

        dict['scaled_lattice'] = np.concatenate([lengths, angles])


class StandardScalerTorch(object):
    """Normalizes the targets of a dataset."""

    def __init__(self, means=None, stds=None):
        self.means = means
        self.stds = stds

    def fit(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)
        self.means = torch.mean(X, dim=0)
        # https://github.com/pytorch/pytorch/issues/29372
        self.stds = torch.std(X, dim=0, unbiased=False) + EPSILON

    def transform(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)
        return (X - self.means) / self.stds

    def inverse_transform(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)
        return X * self.stds + self.means

    def match_device(self, tensor):
        if self.means.device != tensor.device:
            self.means = self.means.to(tensor.device)
            self.stds = self.stds.to(tensor.device)

    def copy(self):
        return StandardScalerTorch(
            means=self.means.clone().detach(),
            stds=self.stds.clone().detach())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"means: {self.means.tolist()}, "
            f"stds: {self.stds.tolist()})"
        )

def get_scaler_from_data_list(data_list, key, pred_node_level=False):
    if pred_node_level:
        data = np.concatenate([d[key] for d in data_list])
    else:
        data = np.array([d[key] for d in data_list])
    print(f"Scaling property {key} with shape {data.shape}")
    targets = torch.tensor(data)
    scaler = StandardScalerTorch()
    scaler.fit(targets)
    return scaler

def lattice_params_to_matrix_torch(lengths, angles):
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    """
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1., 1.)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack([
        lengths[:, 0] * sins[:, 1],
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 0] * coses[:, 1]], dim=1)
    vector_b = torch.stack([
        -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
        lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
        lengths[:, 1] * coses[:, 0]], dim=1)
    vector_c = torch.stack([
        torch.zeros(lengths.size(0), device=lengths.device),
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 2]], dim=1)

    return torch.stack([vector_a, vector_b, vector_c], dim=1)