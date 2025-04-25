from typing import Any, Dict

import math
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from dosmatgen.utils.constants import MAX_ATOMIC_NUM
from dosmatgen.models.cspnet_cfg import CSPNet
from dosmatgen.utils.data import lattice_params_to_matrix_torch
from dosmatgen.utils.diffusion import (
    BetaScheduler, 
    SigmaScheduler, 
    d_log_p_wrapped_normal
)

class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer_name = self.hparams.optim.optimizer
        optimizer = getattr(torch.optim, optimizer_name)(
            self.parameters(), **self.hparams.optim.params
        )

        if not self.hparams.optim.lr_scheduler.use_lr_scheduler:
            return [optimizer]
        
        scheduler_name = self.hparams.optim.lr_scheduler.scheduler
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(
            optimizer, **self.hparams.optim.lr_scheduler.params
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.hparams.optim.lr_scheduler.monitor_metric,
                "strict": False
            },
        }

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class CSPDiffusion(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.decoder = CSPNet(**self.hparams.diffusion.model)
        self.beta_scheduler = BetaScheduler(
            timesteps=self.hparams.diffusion.timesteps, 
            **self.hparams.diffusion.beta_scheduler
        )
        self.sigma_scheduler = SigmaScheduler(
            timesteps=self.hparams.diffusion.timesteps,
            **self.hparams.diffusion.sigma_scheduler
        )

        self.time_dim = self.hparams.diffusion.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_coords = self.hparams.diffusion.cost_coord < 1e-5
        self.keep_lattice = self.hparams.diffusion.cost_lattice < 1e-5

    def forward(self, batch):
        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta = self.beta_scheduler.betas[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)

        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.

        gt_atom_types_onehot = F.one_hot(batch.atom_types-1, num_classes=MAX_ATOMIC_NUM).float()
        rand_t = torch.randn_like(gt_atom_types_onehot)

        c0_repeated = c0.repeat_interleave(batch.num_atoms)[:, None]
        c1_repeated = c1.repeat_interleave(batch.num_atoms)[:, None]
        atom_type_probs = c0_repeated * gt_atom_types_onehot + c1_repeated * rand_t

        if self.keep_coords:
            input_frac_coords = frac_coords
        if self.keep_lattice:
            input_lattice = lattices

        pred_x, pred_l, pred_t, _, _ = self.decoder(
            time_emb, 
            atom_type_probs, 
            input_frac_coords, 
            input_lattice, 
            batch.num_atoms, 
            batch.batch,
            batch.y
        )

        tar_x = d_log_p_wrapped_normal(
            sigmas_per_atom * rand_x, sigmas_per_atom
        ) / torch.sqrt(sigmas_norm_per_atom)

        loss_lattice = F.mse_loss(pred_l, rand_l)
        loss_coord = F.mse_loss(pred_x, tar_x)
        loss_type = F.mse_loss(pred_t, rand_t)

        loss = (
            self.hparams.diffusion.cost_lattice * loss_lattice +
            self.hparams.diffusion.cost_coord * loss_coord + 
            self.hparams.diffusion.cost_type * loss_type
        )

        return {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord,
            'loss_type' : loss_type
        }

    @torch.no_grad()
    def sample(self, batch, diff_ratio=1.0, step_lr=1e-5, unconditional=False, conditional=False):
        batch_size = batch.num_graphs

        l_T = torch.randn([batch_size, 3, 3]).to(self.device)
        x_T = torch.rand([batch.num_nodes, 3]).to(self.device)
        t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device)

        if diff_ratio < 1:
            time_start = int(self.beta_scheduler.timesteps * diff_ratio)
            lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
            atom_types_onehot = F.one_hot(batch.atom_types-1, num_classes=MAX_ATOMIC_NUM).float()
            frac_coords = batch.frac_coords
            rand_l = torch.randn_like(lattices)
            rand_x = torch.randn_like(frac_coords)
            rand_t = torch.randn_like(atom_types_onehot)

            alphas_cumprod = self.beta_scheduler.alphas_cumprod[time_start]
            beta = self.beta_scheduler.betas[time_start]
            c0 = torch.sqrt(alphas_cumprod)
            c1 = torch.sqrt(1. - alphas_cumprod)
            sigmas = self.sigma_scheduler.sigmas[time_start]
            l_T = c0 * lattices + c1 * rand_l
            x_T = (frac_coords + sigmas * rand_x) % 1.
            t_T = c0 * atom_types_onehot + c1 * rand_t
        else:
            time_start = self.beta_scheduler.timesteps

        if self.keep_coords:
            x_T = batch.frac_coords
        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        
        traj = {
            time_start: {
                'num_atoms' : batch.num_atoms,
                'atom_types' : t_T,
                'frac_coords' : x_T % 1.,
                'lattices' : l_T
            }
        }

        for t in tqdm(range(time_start, 0, -1)):
            times = torch.full((batch_size,), t, device=self.device)
            time_emb = self.time_embedding(times)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']
            t_t = traj[t]['atom_types']

            if self.keep_coords:
                x_t = x_T
            if self.keep_lattice:
                l_t = l_T

            # Corrector
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            pred_x, pred_l, pred_t, _, _, = self.decoder(
                time_emb, 
                t_t, 
                x_t, 
                l_t, 
                batch.num_atoms, 
                batch.batch,
                batch.y,
                unconditional=unconditional,
                conditional=conditional
            )

            pred_x = pred_x * torch.sqrt(sigma_norm)
            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_05 = l_t
            t_t_minus_05 = t_t

            # Predictor
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            pred_x, pred_l, pred_t, _, _= self.decoder(
                time_emb, 
                t_t_minus_05, 
                x_t_minus_05, 
                l_t_minus_05, 
                batch.num_atoms, 
                batch.batch,
                batch.y,
                unconditional=unconditional,
                conditional=conditional
            )

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t
            t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) + sigmas * rand_t

            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : t_t_minus_1,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : torch.stack([traj[i]['atom_types'] for i in range(time_start, -1, -1)]).argmax(dim=-1) + 1,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        res = traj[0]
        res['atom_types'] = res['atom_types'].argmax(dim=-1) + 1

        return traj[0], traj_stack
    
    @torch.no_grad()
    def masked_sample(self, batch, diff_ratio=1.0, step_lr=1e-5, mask=None):
        batch_size = batch.num_graphs

        l_T = torch.randn([batch_size, 3, 3]).to(self.device)
        x_T = torch.rand([batch.num_nodes, 3]).to(self.device)
        t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device)

        if self.keep_coords:
            x_T = batch.frac_coords
        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        
        traj = {
            self.beta_scheduler.timesteps: {
                'num_atoms' : batch.num_atoms,
                'atom_types' : t_T,
                'frac_coords' : x_T % 1.,
                'lattices' : l_T
            }
        }

        for t in tqdm(range(self.beta_scheduler.timesteps, 0, -1)):
            times = torch.full((batch_size,), t, device=self.device)
            time_emb = self.time_embedding(times)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']
            t_t = traj[t]['atom_types']

            if self.keep_coords:
                x_t = x_T
            if self.keep_lattice:
                l_t = l_T

            # Corrector
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            y = batch.y * mask

            pred_x, pred_l, pred_t, _, _, = self.decoder(
                time_emb, 
                t_t, 
                x_t, 
                l_t, 
                batch.num_atoms, 
                batch.batch,
                y
            )

            pred_x = pred_x * torch.sqrt(sigma_norm)
            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_05 = l_t
            t_t_minus_05 = t_t

            # Predictor
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            pred_x, pred_l, pred_t, _, _= self.decoder(
                time_emb, 
                t_t_minus_05, 
                x_t_minus_05, 
                l_t_minus_05, 
                batch.num_atoms, 
                batch.batch,
                y
            )

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t
            t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) + sigmas * rand_t

            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : t_t_minus_1,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : torch.stack([traj[i]['atom_types'] for i in range(self.beta_scheduler.timesteps, -1, -1)]).argmax(dim=-1) + 1,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(self.beta_scheduler.timesteps, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(self.beta_scheduler.timesteps, -1, -1)])
        }

        res = traj[0]
        res['atom_types'] = res['atom_types'].argmax(dim=-1) + 1

        return traj[0], traj_stack

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_type = output_dict['loss_type']
        loss = output_dict['loss']

        self.log_dict(
            {
                'train_loss': loss,
                'lattice_loss': loss_lattice,
                'coord_loss': loss_coord,
                'type_loss': loss_type
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )

        if loss.isnan():
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, output_dict, prefix):
        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_type = output_dict['loss_type']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord,
            f'{prefix}_type_loss': loss_type,
        }
        return log_dict, loss
    
    @torch.no_grad()
    def cfg_sample(self, batch, diff_ratio=1.0, step_lr=1e-5, w=1.0):
        batch_size = batch.num_graphs

        l_T = torch.randn([batch_size, 3, 3]).to(self.device)
        x_T = torch.rand([batch.num_nodes, 3]).to(self.device)
        t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device)

        if diff_ratio < 1:
            time_start = int(self.beta_scheduler.timesteps * diff_ratio)
            lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
            atom_types_onehot = F.one_hot(batch.atom_types-1, num_classes=MAX_ATOMIC_NUM).float()
            frac_coords = batch.frac_coords
            rand_l = torch.randn_like(lattices)
            rand_x = torch.randn_like(frac_coords)
            rand_t = torch.randn_like(atom_types_onehot)

            alphas_cumprod = self.beta_scheduler.alphas_cumprod[time_start]
            beta = self.beta_scheduler.betas[time_start]
            c0 = torch.sqrt(alphas_cumprod)
            c1 = torch.sqrt(1. - alphas_cumprod)
            sigmas = self.sigma_scheduler.sigmas[time_start]
            l_T = c0 * lattices + c1 * rand_l
            x_T = (frac_coords + sigmas * rand_x) % 1.
            t_T = c0 * atom_types_onehot + c1 * rand_t
        else:
            time_start = self.beta_scheduler.timesteps

        if self.keep_coords:
            x_T = batch.frac_coords
        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        
        traj = {
            time_start: {
                'num_atoms' : batch.num_atoms,
                'atom_types' : t_T,
                'frac_coords' : x_T % 1.,
                'lattices' : l_T
            }
        }

        for t in tqdm(range(time_start, 0, -1)):
            times = torch.full((batch_size,), t, device=self.device)
            time_emb = self.time_embedding(times)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']
            t_t = traj[t]['atom_types']

            if self.keep_coords:
                x_t = x_T
            if self.keep_lattice:
                l_t = l_T

            # Corrector
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            # unconditional sample
            uncod_pred_x, uncod_pred_l, uncod_pred_t, _, _, = self.decoder(
                time_emb, 
                t_t, 
                x_t, 
                l_t, 
                batch.num_atoms, 
                batch.batch,
                batch.y,
                unconditional=True,
                conditional=False
            )

            # conditional sample
            cod_pred_x, cod_pred_l, cod_pred_t, _, _= self.decoder(
                time_emb, 
                t_t, 
                x_t, 
                l_t, 
                batch.num_atoms, 
                batch.batch,
                batch.y,
                unconditional=False,
                conditional=True
            )

            pred_x = (1-w) * cod_pred_x + w * uncod_pred_x
            pred_l = (1-w) * cod_pred_l + w * uncod_pred_l
            pred_t = (1-w) * cod_pred_t + w * uncod_pred_t

            # pred_x = (1+w) * uncod_pred_x - w * cod_pred_x
            # pred_l = (1+w) * uncod_pred_l - w * cod_pred_l
            # pred_t = (1+w) * uncod_pred_t - w * cod_pred_t

            # pred_x = (1+w) * cod_pred_x - w * uncod_pred_x
            # pred_l = (1+w) * cod_pred_l - w * uncod_pred_l
            # pred_t = (1+w) * cod_pred_t - w * uncod_pred_t

            pred_x = pred_x * torch.sqrt(sigma_norm)
            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_05 = l_t
            t_t_minus_05 = t_t

            # Predictor
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            # unconditional sample
            uncod_pred_x, uncod_pred_l, uncod_pred_t, _, _, = self.decoder(
                time_emb, 
                t_t, 
                x_t, 
                l_t, 
                batch.num_atoms, 
                batch.batch,
                batch.y,
                unconditional=True,
                conditional=False
            )

            # conditional sample
            cod_pred_x, cod_pred_l, cod_pred_t, _, _= self.decoder(
                time_emb, 
                t_t, 
                x_t, 
                l_t, 
                batch.num_atoms, 
                batch.batch,
                batch.y,
                unconditional=False,
                conditional=True
            )

            pred_x = (1-w) * cod_pred_x + w * uncod_pred_x
            pred_l = (1-w) * cod_pred_l + w * uncod_pred_l
            pred_t = (1-w) * cod_pred_t + w * uncod_pred_t

            # pred_x = (1+w) * uncod_pred_x - w * cod_pred_x
            # pred_l = (1+w) * uncod_pred_l - w * cod_pred_l
            # pred_t = (1+w) * uncod_pred_t - w * cod_pred_t

            # pred_x = (1+w) * cod_pred_x - w * uncod_pred_x
            # pred_l = (1+w) * cod_pred_l - w * uncod_pred_l
            # pred_t = (1+w) * cod_pred_t - w * uncod_pred_t

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t
            t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) + sigmas * rand_t

            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : t_t_minus_1,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : torch.stack([traj[i]['atom_types'] for i in range(time_start, -1, -1)]).argmax(dim=-1) + 1,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        res = traj[0]
        res['atom_types'] = res['atom_types'].argmax(dim=-1) + 1

        return traj[0], traj_stack
    
    @torch.no_grad()
    def fix_sample(self, batch, diff_ratio=1.0, step_lr=1e-5, w=1.0, fix_atom_type=28):
        batch_size = batch.num_graphs

        l_T = torch.randn([batch_size, 3, 3]).to(self.device)
        x_T = torch.rand([batch.num_nodes, 3]).to(self.device)
        t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device)

        fix_t_T = F.one_hot(torch.tensor([fix_atom_type-1], device=self.device), num_classes=MAX_ATOMIC_NUM).float()

        if diff_ratio < 1:
            time_start = int(self.beta_scheduler.timesteps * diff_ratio)
            lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
            atom_types_onehot = F.one_hot(batch.atom_types-1, num_classes=MAX_ATOMIC_NUM).float()
            frac_coords = batch.frac_coords
            rand_l = torch.randn_like(lattices)
            rand_x = torch.randn_like(frac_coords)
            rand_t = torch.randn_like(atom_types_onehot)

            alphas_cumprod = self.beta_scheduler.alphas_cumprod[time_start]
            beta = self.beta_scheduler.betas[time_start]
            c0 = torch.sqrt(alphas_cumprod)
            c1 = torch.sqrt(1. - alphas_cumprod)
            sigmas = self.sigma_scheduler.sigmas[time_start]
            l_T = c0 * lattices + c1 * rand_l
            x_T = (frac_coords + sigmas * rand_x) % 1.
            t_T = c0 * atom_types_onehot + c1 * rand_t
        else:
            time_start = self.beta_scheduler.timesteps

        if self.keep_coords:
            x_T = batch.frac_coords
        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        
        # fix atom type at position 0
        t_T[0] = fix_t_T.detach().clone()
        
        traj = {
            time_start: {
                'num_atoms' : batch.num_atoms,
                'atom_types' : t_T,
                'frac_coords' : x_T % 1.,
                'lattices' : l_T
            }
        }

        for t in tqdm(range(time_start, 0, -1)):
            times = torch.full((batch_size,), t, device=self.device)
            time_emb = self.time_embedding(times)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']
            t_t = traj[t]['atom_types']

            if self.keep_coords:
                x_t = x_T
            if self.keep_lattice:
                l_t = l_T

            # fix atom type at position 0
            t_t[0] = fix_t_T.detach().clone()

            # Corrector
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            # unconditional sample
            uncod_pred_x, uncod_pred_l, uncod_pred_t, _, _, = self.decoder(
                time_emb, 
                t_t, 
                x_t, 
                l_t, 
                batch.num_atoms, 
                batch.batch,
                batch.y,
                unconditional=True,
                conditional=False
            )

            # conditional sample
            cod_pred_x, cod_pred_l, cod_pred_t, _, _= self.decoder(
                time_emb, 
                t_t, 
                x_t, 
                l_t, 
                batch.num_atoms, 
                batch.batch,
                batch.y,
                unconditional=False,
                conditional=True
            )

            pred_x = (1-w) * cod_pred_x + w * uncod_pred_x
            pred_l = (1-w) * cod_pred_l + w * uncod_pred_l
            pred_t = (1-w) * cod_pred_t + w * uncod_pred_t

            # pred_x = (1+w) * uncod_pred_x - w * cod_pred_x
            # pred_l = (1+w) * uncod_pred_l - w * cod_pred_l
            # pred_t = (1+w) * uncod_pred_t - w * cod_pred_t

            # pred_x = (1+w) * cod_pred_x - w * uncod_pred_x
            # pred_l = (1+w) * cod_pred_l - w * uncod_pred_l
            # pred_t = (1+w) * cod_pred_t - w * uncod_pred_t

            pred_x = pred_x * torch.sqrt(sigma_norm)
            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_05 = l_t
            t_t_minus_05 = t_t

            # Predictor
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            # unconditional sample
            uncod_pred_x, uncod_pred_l, uncod_pred_t, _, _, = self.decoder(
                time_emb, 
                t_t, 
                x_t, 
                l_t, 
                batch.num_atoms, 
                batch.batch,
                batch.y,
                unconditional=True,
                conditional=False
            )

            # conditional sample
            cod_pred_x, cod_pred_l, cod_pred_t, _, _= self.decoder(
                time_emb, 
                t_t, 
                x_t, 
                l_t, 
                batch.num_atoms, 
                batch.batch,
                batch.y,
                unconditional=False,
                conditional=True
            )

            pred_x = (1-w) * cod_pred_x + w * uncod_pred_x
            pred_l = (1-w) * cod_pred_l + w * uncod_pred_l
            pred_t = (1-w) * cod_pred_t + w * uncod_pred_t

            # pred_x = (1+w) * uncod_pred_x - w * cod_pred_x
            # pred_l = (1+w) * uncod_pred_l - w * cod_pred_l
            # pred_t = (1+w) * uncod_pred_t - w * cod_pred_t

            # pred_x = (1+w) * cod_pred_x - w * uncod_pred_x
            # pred_l = (1+w) * cod_pred_l - w * uncod_pred_l
            # pred_t = (1+w) * cod_pred_t - w * uncod_pred_t

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t
            t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) + sigmas * rand_t

            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : t_t_minus_1,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : torch.stack([traj[i]['atom_types'] for i in range(time_start, -1, -1)]).argmax(dim=-1) + 1,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        res = traj[0]
        res['atom_types'] = res['atom_types'].argmax(dim=-1) + 1

        return traj[0], traj_stack