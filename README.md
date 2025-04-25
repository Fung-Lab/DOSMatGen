# DOSMatGen
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2504.06249)
Electronic Structure Guided Inverse Design Using Generative Models

## Environment
```
# create a conda environment
conda create -n dosmatgen python=3.12
conda activate dosmatgen

# pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# pytorch geometric
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

# install additional dependencies
pip install ase pymatgen lightning p-tqdm omegaconf einops wandb 

# ipynb support
pip install ipykernel ipywidgets
```

## Dataset
The datamodule expects crystal data in JSON format, structured as a list of dictionaries with the following keys:
```
[
    ...
    {
        'structure_id'      : str,
        'positions'         : list of (N, 3),
        'cell'              : list of (3, 3),
        'atomic_numbers'    : list of (N),
        'y'                 : list of (N, M)
    }
    ...
]
```
where `y` is the node-level features.

See below on how to download the MP-DOS dataset.

## Sample Run
To train a diffusion model with CFG (classifier-free guidance) on the MP-DOS dataset, first take a look at the config file `configs/dos_cfg.yml`, then run
```
python run_diffusion_CFG.py
```
The training progress is tracked using `wandb`. If you're running it for the first time, you may need to sign in.


## Generation
Since we are using classifier-free guidance, we are still able to do unconditional generation, that is, generating structures without input DOS information. To do so, try
```
python generate_CFG_unconditional.py --root_path=<PATH/TO/CKPT>
```

For conditional generation, we need to feed the model the DOS information we want to condition the generation on. As an example, we can try to reconstruct the MP-DOS test split:

```
python generate_CFG_conditional.py --root_path=<PATH/TO/CKPT>
```

Generated structures will be saved to a timestamped folder in `structures/`.

## Visualization
Please refer to `plot_dos.ipynb` on a possible approach to visualize the generated structures' DOS against the ground truth.

## Download Model Weights and Dataset
Two trained model weights, (1) diffusion model with CFG and (2) DOS forward model, can be download at the following link:
https://u.pcloud.link/publink/show?code=XZ0kE45ZOuEPUrHzQU5ciR2pXKCkDzNY3F9k

The MP-DOS dataset is available at the following link:
https://u.pcloud.link/publink/show?code=XZ4kE45ZhFgF7IWysbuKnuUExSsnfLC2n9wy

The default directory structure for this repo is:

```
DOSMatGen
├── configs
├── data
│   └── mp_dos
├── outputs
│   ├── dos_cfg
│   └── dos_forward_model
├── dosmatgen
│   ├── dataset
│   ├── diffusion
│   ├── models
│   └── utils
└── structures
    └── 08032025_160513
```

## Citation
```
@article{jia2025electronic,
  title={Electronic Structure Guided Inverse Design Using Generative Models},
  author={Jia, Shuyi and Ganesh, Panchapakesan and Fung, Victor},
  journal={arXiv preprint arXiv:2504.06249},
  year={2025}
}
```