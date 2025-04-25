from typing import Any

import math
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

import pytorch_lightning as pl

from dosmatgen.utils.constants import MAX_ATOMIC_NUM
from dosmatgen.models.cspnet import CSPNet
from dosmatgen.utils.data import lattice_params_to_matrix_torch
from dosmatgen.utils.diffusion import (
    BetaScheduler, 
    SigmaScheduler
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
    
def judge_requires_grad(obj):
    if isinstance(obj, torch.Tensor):
        return obj.requires_grad
    elif isinstance(obj, nn.Module):
        return next(obj.parameters()).requires_grad
    else:
        raise TypeError
    
class RequiresGradContext(object):
    def __init__(self, *objs, requires_grad):
        self.objs = objs
        self.backups = [judge_requires_grad(obj) for obj in objs]
        if isinstance(requires_grad, bool):
            self.requires_grads = [requires_grad] * len(objs)
        elif isinstance(requires_grad, list):
            self.requires_grads = requires_grad
        else:
            raise TypeError
        assert len(self.objs) == len(self.requires_grads)

    def __enter__(self):
        for obj, requires_grad in zip(self.objs, self.requires_grads):
            obj.requires_grad_(requires_grad)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for obj, backup in zip(self.objs, self.backups):
            obj.requires_grad_(backup)

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

class CSPProperty(BaseModule):
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
        self.time_independent = self.hparams.diffusion.get('time_independent', False)
        print(f"Time independence: {self.time_independent}")

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

        rand_l = torch.randn_like(lattices)
        rand_x = torch.randn_like(frac_coords)

        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.

        gt_atom_types_onehot = F.one_hot(batch.atom_types-1, num_classes=MAX_ATOMIC_NUM).float()
        rand_t = torch.randn_like(gt_atom_types_onehot)

        c0_repeated = c0.repeat_interleave(batch.num_atoms)[:, None]
        c1_repeated = c1.repeat_interleave(batch.num_atoms)[:, None]
        atom_type_probs = c0_repeated * gt_atom_types_onehot + c1_repeated * rand_t

        # time independence
        if self.time_independent:
            time_emb = torch.zeros_like(time_emb)
            atom_type_probs = gt_atom_types_onehot
            input_frac_coords = frac_coords
            input_lattice = lattices

        pred_x, pred_l, pred_t, pred_graph, pred_node = self.decoder(
            time_emb, 
            atom_type_probs, 
            input_frac_coords, 
            input_lattice, 
            batch.num_atoms, 
            batch.batch
        )

        if self.decoder.pred_graph_level:
            loss = F.l1_loss(pred_graph, batch.y)
        elif self.decoder.pred_node_level:
            loss = F.l1_loss(pred_node, batch.y)
        else:
            raise ValueError("Invalid prediction level")

        return {
            'loss' : loss
        }

    @torch.no_grad()
    def infer(self, batch):
        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        time_emb = torch.zeros_like(time_emb)
        atom_type_probs = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()
        input_frac_coords = batch.frac_coords
        input_lattice = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        pred_x, pred_l, pred_t, pred_graph, pred_node = self.decoder(
            time_emb,
            atom_type_probs,
            input_frac_coords,
            input_lattice,
            batch.num_atoms,
            batch.batch
        )

        if self.decoder.pred_graph_level:
            return pred_graph, batch.y
        elif self.decoder.pred_node_level:
            return pred_node, batch.y
        else:
            raise ValueError("Invalid prediction level")

    @torch.no_grad()
    def sample(self, batch, uncod, diff_ratio=1.0, step_lr=1e-5, aug=1.0):
        assert self.time_independent == False, "Time independence is not supported for denoising; use self.infer()"

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

            if self.hparams.diffusion.latent_dim > 0:            
                raise NotImplementedError

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)
            c2 = (1 - alphas) / torch.sqrt(alphas)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']
            t_t = traj[t]['atom_types']

            # Corrector
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            pred_x, pred_l, pred_t, _, _, = uncod.decoder(
                time_emb, 
                t_t, 
                x_t, 
                l_t, 
                batch.num_atoms, 
                batch.batch
            )

            pred_x = pred_x * torch.sqrt(sigma_norm)
            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x
            l_t_minus_05 = l_t
            t_t_minus_05 = t_t

            # Predictor
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))

            pred_x, pred_l, pred_t, _, _, = uncod.decoder(
                time_emb, 
                t_t_minus_05, 
                x_t_minus_05, 
                l_t_minus_05, 
                batch.num_atoms, 
                batch.batch
            )

            with torch.enable_grad():
                with RequiresGradContext(t_t_minus_05, x_t_minus_05, l_t_minus_05, requires_grad=True):
                    _, _, _, pred_graph, pred_node = self.decoder(
                        time_emb, 
                        t_t_minus_05, 
                        x_t_minus_05, 
                        l_t_minus_05, 
                        batch.num_atoms, 
                        batch.batch
                    )

                    if self.decoder.pred_graph_level:
                        val = torch.linalg.norm(pred_graph - batch.y, dim=1, keepdim=True)
                    elif self.decoder.pred_node_level:
                        val = torch.linalg.norm(pred_node - batch.y, dim=1, keepdim=True)
                    else:
                        raise ValueError("Invalid prediction level")
                    
                    grad_outputs = [torch.ones_like(val)]
                    grad_t, grad_x, grad_l = grad(
                        val, 
                        [t_t_minus_05, x_t_minus_05, l_t_minus_05], 
                        grad_outputs=grad_outputs, 
                        allow_unused=True
                    )

            pred_x = pred_x * torch.sqrt(sigma_norm)
            x_t_minus_1 = x_t_minus_05 - step_size * pred_x - (std_x ** 2) * aug * grad_x + std_x * rand_x 
            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) - (sigmas ** 2) * aug * grad_l + sigmas * rand_l 
            t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) - (sigmas ** 2) * aug * grad_t + sigmas * rand_t

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
    def masked_sample(self, batch, uncod, diff_ratio=1.0, step_lr=1e-5, aug=1.0, mask=None):
        assert self.time_independent == False, "Time independence is not supported for denoising; use self.infer()"

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

            if self.hparams.diffusion.latent_dim > 0:            
                raise NotImplementedError

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)
            c2 = (1 - alphas) / torch.sqrt(alphas)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']
            t_t = traj[t]['atom_types']

            # Corrector
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            pred_x, pred_l, pred_t, _, _, = uncod.decoder(
                time_emb, 
                t_t, 
                x_t, 
                l_t, 
                batch.num_atoms, 
                batch.batch
            )

            pred_x = pred_x * torch.sqrt(sigma_norm)
            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x
            l_t_minus_05 = l_t
            t_t_minus_05 = t_t

            # Predictor
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))

            pred_x, pred_l, pred_t, _, _, = uncod.decoder(
                time_emb, 
                t_t_minus_05, 
                x_t_minus_05, 
                l_t_minus_05, 
                batch.num_atoms, 
                batch.batch
            )

            with torch.enable_grad():
                with RequiresGradContext(t_t_minus_05, x_t_minus_05, l_t_minus_05, requires_grad=True):
                    _, _, _, pred_graph, pred_node = self.decoder(
                        time_emb, 
                        t_t_minus_05, 
                        x_t_minus_05, 
                        l_t_minus_05, 
                        batch.num_atoms, 
                        batch.batch
                    )

                    if self.decoder.pred_graph_level:
                        val = torch.linalg.norm(pred_graph - batch.y, dim=1, keepdim=True)
                    elif self.decoder.pred_node_level:
                        val = torch.linalg.norm(pred_node - batch.y, dim=1, keepdim=True)
                    else:
                        raise ValueError("Invalid prediction level")
                    
                    grad_outputs = [torch.ones_like(val)]
                    grad_t, grad_x, grad_l = grad(
                        val, 
                        [t_t_minus_05, x_t_minus_05, l_t_minus_05], 
                        grad_outputs=grad_outputs, 
                        allow_unused=True
                    )
            
            assert len(mask) == grad_x.shape[0] == grad_t.shape[0]
            grad_x = grad_x * mask
            grad_t = grad_t * mask

            pred_x = pred_x * torch.sqrt(sigma_norm)
            x_t_minus_1 = x_t_minus_05 - step_size * pred_x - (std_x ** 2) * aug * grad_x + std_x * rand_x 
            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l 
            t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) - (sigmas ** 2) * aug * grad_t + sigmas * rand_t

            # x_t_minus_1 = x_t_minus_05 - step_size * pred_x - (std_x ** 2) * aug * grad_x + std_x * rand_x 
            # l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) - (sigmas ** 2) * aug * grad_l + sigmas * rand_l 
            # t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) - (sigmas ** 2) * aug * grad_t + sigmas * rand_t

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

    def multinomial_sample(self, t_t, pred_t, num_atoms, times):
        noised_atom_types = t_t
        pred_atom_probs = F.softmax(pred_t, dim=-1)

        alpha = self.beta_scheduler.alphas[times].repeat_interleave(num_atoms)
        alpha_bar = self.beta_scheduler.alphas_cumprod[times-1].repeat_interleave(num_atoms)

        theta = (alpha[:, None] * noised_atom_types + (1 - alpha[:, None]) / MAX_ATOMIC_NUM) * \
                (alpha_bar[:, None] * pred_atom_probs + (1 - alpha_bar[:, None]) / MAX_ATOMIC_NUM)
        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)
        return theta

    def type_loss(self, pred_atom_types, target_atom_types, noised_atom_types, batch, times):
        pred_atom_probs = F.softmax(pred_atom_types, dim=-1)
        atom_probs_0 = F.one_hot(target_atom_types-1, num_classes=MAX_ATOMIC_NUM)

        alpha = self.beta_scheduler.alphas[times].repeat_interleave(batch.num_atoms)
        alpha_bar = self.beta_scheduler.alphas_cumprod[times-1].repeat_interleave(batch.num_atoms)

        theta = (alpha[:, None] * noised_atom_types + (1 - alpha[:, None]) / MAX_ATOMIC_NUM) * \
                (alpha_bar[:, None] * atom_probs_0 + (1 - alpha_bar[:, None]) / MAX_ATOMIC_NUM)
        theta_hat = (alpha[:, None] * noised_atom_types + (1 - alpha[:, None]) / MAX_ATOMIC_NUM) * \
                    (alpha_bar[:, None] * pred_atom_probs + (1 - alpha_bar[:, None]) / MAX_ATOMIC_NUM)

        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)
        theta_hat = theta_hat / (theta_hat.sum(dim=-1, keepdim=True) + 1e-8)
        theta_hat = torch.log(theta_hat + 1e-8)

        kldiv = F.kl_div(
            input=theta_hat, 
            target=theta, 
            reduction='none',
            log_target=False
        ).sum(dim=-1)

        return kldiv.mean()

    def lap(self, probs, types, num_atoms):
        types_1 = types - 1
        atoms_end = torch.cumsum(num_atoms, dim=0)
        atoms_begin = torch.zeros_like(num_atoms)
        atoms_begin[1:] = atoms_end[:-1]
        res_types = []
        for st, ed in zip(atoms_begin, atoms_end):
            types_crys = types_1[st:ed]
            probs_crys = probs[st:ed]
            probs_crys = probs_crys[:,types_crys]
            probs_crys = F.softmax(probs_crys, dim=-1).detach().cpu().numpy()
            assignment = linear_sum_assignment(-probs_crys)[1].astype(np.int32)
            types_crys = types_crys[assignment] + 1
            res_types.append(types_crys)
        return torch.cat(res_types)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch)
        loss = output_dict['loss']

        self.log_dict(
            {
                'train_loss': loss
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
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss
        }
        return log_dict, loss