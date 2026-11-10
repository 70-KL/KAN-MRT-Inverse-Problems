#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KAN–SRT (BGK) viscosity-inverse ablation for Re=1000 lid-driven cavity.

Single run trains ONLY this model; no MRT training.
Uses EXACTLY the two-branch KANLinear form of MRT-in-nu-Z-all-1000(1).py:
  KAN_eq:  (x,y) -> (rho,u,v)    [2,30,3]
  KAN_neq: (x,y) -> 9 fneq       [2,30,9]

PDE (the original source's steady continuous-space representation):
  R_i = xi_i . grad( f_i^eq + f_i^neq ) + f_i^neq/tau_hat,
  tau_hat = 3*nu_hat + 0.5.

BOUNDARY (selected SRT first-order CE-like closure, NOT a fixed true tau):
  f_i^{neq,*} = -tau_hat (xi_i . grad f_i^eq),  for selected wall directions;
  R_i|boundary = xi_i . grad f_i + f_i^neq/tau_hat for remaining directions.
The same LEARNABLE tau_hat is used in PDE and both boundary constraints.
No reference viscosity enters any training loss. True nu=0.001 only evaluates error.

Keep caveats visible in the paper: the code follows the source's D2Q9
continuous spatial residual with discrete-time tau(nu) mapping. Its
first-order boundary CE approximation omits the temporal derivative
because the cavity field is steady. The original code's wall-direction
classification is retained for source comparability, and the separately
predicted fneq moments are not projected to exact mass/momentum conservation.
Changing the MRT boundary formula to CE as well as collision changes more
than one item: a strictly isolated MRT-vs-SRT comparison must harmonize
boundary treatment or describe this confound explicitly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # ✅ 防止OpenMP库冲突
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

SCRIPT_DIR = Path(__file__).resolve().parent
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_torch(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
                .expand(in_features, -1)
                .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )

class KAN_eq(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=15,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[0.0, 1.0],
        initial_nu_hat=0.01,
    ):
        super(KAN_eq, self).__init__()

        self.nu_hat = nn.Parameter(torch.tensor([initial_nu_hat], requires_grad=True).float().to(device))
        print(f"    可学习参数: 黏度 nu_hat (初始值: {initial_nu_hat:.4f})")
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
        self.history = {
            'total_loss': [], 'loss_pde': [], 'loss_bc': [], 'loss_data': [], 'nu_hat': []  # 新增项
        }
        print("  - 逆问题模型初始化完成。")

    def forward(self, x: torch.Tensor, update_grid=False):
        for index, layer in enumerate(self.layers):
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
            if index < len(self.layers)-1: # 如果不是最后一层，拉到-1到1之间，最后一层不需要tanh
                x = torch.tanh(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
    def get_tau(self):
        return 3.0 * self.nu_hat + 0.5  # 修正：使用self.nu_hat

class KAN_neq(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=15,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[0.0, 1.0],
    ):
        super(KAN_neq, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()

        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for index, layer in enumerate(self.layers):
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
            if index < len(self.layers)-1: # 如果不是最后一层，拉到-1到1之间，最后一层不需要tanh
                x = torch.tanh(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )



# -----------------------------------------------------------------------------
# Configuration & data generation
# -----------------------------------------------------------------------------
@dataclass
class Config:
    data: str = str(SCRIPT_DIR / 'num-1000.csv')
    output: str = ''
    seed: int = 42
    data_seed: int = 2026
    initial_nu: float = -1.0  # negative sentinel means original-style U(0,0.1)
    true_nu: float = 0.001  # evaluation ONLY; never used in loss unless legacy fixed tau
    grid_size: int = 15
    spline_order: int = 3
    width: int = 30
    epochs: int = 80000
    n_obs: int = 1000
    n_f: int = 20000
    n_b: int = 5000  # PER WALL
    batch_f: int = 0   # 0 uses full 20000 for source comparability
    batch_b: int = 0   # 0 uses full 5000 per wall
    lr: float = 1e-3
    lr_nu: float = 5e-4
    w_pde: float = 1.0
    w_data: float = 1.0
    w_bc: float = 1.0
    lbfgs_max_iter: int = 100
    lbfgs_max_eval: int = 200
    print_every: int = 1000
    save_every: int = 20000
    eval_grid_size: int = 100
    plot_dpi: int = 300


def load_data(path, n_obs, seed):
    df = pd.read_csv(path)
    required = {'X','Y','u','v'}
    if not required.issubset(df.columns):
        raise ValueError(f'{path} must contain columns X,Y,u,v; got {df.columns.tolist()}')
    for col in required:
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f'Nonfinite CSV values in {col}')
    n = min(n_obs, len(df))
    rs = np.random.default_rng(seed)
    ids = rs.choice(len(df), n, replace=False)
    return df, ids


def d2q9(device):
    xi = torch.tensor([[0,0],[1,0],[0,1],[-1,0],[0,-1],
                       [1,1],[-1,1],[-1,-1],[1,-1]], dtype=torch.float32, device=device)
    w = torch.tensor([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36],
                     dtype=torch.float32, device=device)
    return xi,w


def make_bounds(n, rng, device):
    # Keep four original lid-driven cavity walls. Corner points are not sampled explicitly.
    arr = []
    for name, side, normal, uv in [
        ('top','top',(0.,1.),(1.,0.)),
        ('left','left',(-1.,0.),(0.,0.)),
        ('right','right',(1.,0.),(0.,0.)),
        ('bottom','bottom',(0.,-1.),(0.,0.))]:
        r = rng.random(n).astype(np.float32)
        if side == 'top': XY = np.column_stack([r,np.ones(n,np.float32)])
        elif side == 'bottom': XY = np.column_stack([r,np.zeros(n,np.float32)])
        elif side == 'left': XY = np.column_stack([np.zeros(n,np.float32),r])
        else: XY = np.column_stack([np.ones(n,np.float32),r])
        U = np.tile(np.asarray(uv,dtype=np.float32), (n,1))
        arr.append(dict(name=name,xy=torch.as_tensor(XY,device=device),
                        uv=torch.as_tensor(U,device=device),
                        normal=torch.tensor(normal,device=device)))
    return arr


def draw_subset(x, count, gen):
    if count <= 0 or count >= len(x):
        return x
    ids = torch.randperm(len(x),device=x.device, generator=gen)[:count]
    return x[ids]


def get_macro_eq(model_eq, X):
    raw = model_eq(X)
    rho,u,v=raw[:,0:1],raw[:,1:2],raw[:,2:3]
    return rho,u,v


def equilibrium(rho,u,v,xi,w):
    vel = torch.cat([u,v],dim=1)
    eu = vel @ xi.T
    return w.unsqueeze(0)*rho*(1.0+3.0*eu+4.5*eu.square()-1.5*(u.square()+v.square()))


def autograd_coords(y,x):
    return torch.autograd.grad(y.sum(),x,create_graph=True,retain_graph=True)[0]


def srt_pde_loss(X,eq_net,neq_net,xi,w):
    X=X.detach().requires_grad_(True)
    rho,u,v=get_macro_eq(eq_net,X)
    feq=equilibrium(rho,u,v,xi,w)
    fneq=neq_net(X)
    f=feq+fneq
    grad_terms=[]
    for i in range(9):
        gradient=autograd_coords(f[:,i:i+1],X)
        grad_terms.append((gradient*xi[i].unsqueeze(0)).sum(dim=1,keepdim=True))
    streaming=torch.cat(grad_terms,dim=1)
    tau=eq_net.get_tau()
    residual=streaming+fneq/tau
    # Source MRT file returns sqrt(mean(R^2)+1e-10), not mean(R^2).
    return torch.sqrt(residual.square().mean()+1e-10)


def data_loss(eq_net,xy,uv):
    raw=eq_net(xy)
    return ((raw[:,1:2]-uv[:,0:1]).square().mean()
           +(raw[:,2:3]-uv[:,1:2]).square().mean())


def boundary_components_one(X,uv,n,eq_net,neq_net,xi,w,cfg):
    X=X.detach().requires_grad_(True)
    rho,u,v=get_macro_eq(eq_net,X)
    mac=(rho-1.0).square().mean()+(u-uv[:,0:1]).square().mean()+(v-uv[:,1:2]).square().mean()
    feq=equilibrium(rho,u,v,xi,w)
    fneq=neq_net(X)
    f=feq+fneq
    tau=eq_net.get_tau()  # learned SRT tau from the SAME nu_hat as the PDE
    # Incoming/outgoing sign convention inherited exactly from supplied source.
    e_dot_n=xi@n
    incoming=[i for i in range(9) if e_dot_n[i]>0]
    outgoing=[i for i in range(9) if e_dot_n[i]<=0]
    fbc=torch.zeros((),device=X.device)
    ebc=torch.zeros((),device=X.device)
    for i in incoming:
        gf=autograd_coords(feq[:,i:i+1],X)
        adv=(gf*xi[i]).sum(dim=1,keepdim=True)
        # First-order Chapman-Enskog-like approximation of steady BGK:
        #   f_i^{neq,*} = -tau_hat * (xi_i . grad(f_i^eq))
        # Use learned tau_hat; never fix tau with ground-truth viscosity.
        target=-tau*adv
        fbc=fbc+(fneq[:,i:i+1]-target).square().mean()
    for j in outgoing:
        gf=autograd_coords(f[:,j:j+1],X)
        adv=(gf*xi[j]).sum(dim=1,keepdim=True)
        r=adv+fneq[:,j:j+1]/tau
        ebc=ebc+r.square().mean()
    return mac,fbc,ebc


def boundary_loss(bcs,eq_net,neq_net,xi,w,cfg,gen):
    mbc=fbc=ebc=torch.zeros((),device=xi.device)
    for wall in bcs:
        if cfg.batch_b>0 and cfg.batch_b<len(wall['xy']):
            ids=torch.randperm(len(wall['xy']),device=xi.device,generator=gen)[:cfg.batch_b]
            xy=wall['xy'][ids];uv=wall['uv'][ids]
        else:
            xy=wall['xy'];uv=wall['uv']
        m,f,e=boundary_components_one(xy,uv,wall['normal'],eq_net,neq_net,xi,w,cfg)
        mbc=mbc+m;fbc=fbc+f;ebc=ebc+e
    return mbc+fbc+ebc,(mbc,fbc,ebc)


def save_history(out,history):
    pd.DataFrame(history).to_csv(out/'loss_history.csv',index=False)
    if history:
        pd.DataFrame([{'phase':e['phase'],'iteration':e['iteration'],
                       'nu_hat':e['nu_hat'],'nu_grad':e['nu_grad']} for e in history]).to_csv(
                       out/'nu_history.csv',index=False)


def infer_macro(eq_net,coords,dev,chunk=2048):
    eq_net.eval();out=[]
    with torch.no_grad():
        for data in np.array_split(coords,max(1,math.ceil(len(coords)/chunk))):
            out.append(eq_net(torch.tensor(data,dtype=torch.float32,device=dev)).cpu().numpy())
    return np.concatenate(out)


def relative_l2(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12))


def save_final_results(cfg,df,ids,eq_net,neq_net,out,dev,history,timing):
    inp=df[['X','Y']].to_numpy(np.float32)
    ref=df[['u','v']].to_numpy(np.float32)
    pred=infer_macro(eq_net,inp,dev)
    rho,u,v=pred[:,0],pred[:,1],pred[:,2]
    vals=pd.DataFrame(dict(X=inp[:,0],Y=inp[:,1],u_ref=ref[:,0],v_ref=ref[:,1],
         rho_pred=rho,u_pred=u,v_pred=v,
         abs_u_error=np.abs(u-ref[:,0]),abs_v_error=np.abs(v-ref[:,1]),
         abs_vector_error=np.hypot(u-ref[:,0],v-ref[:,1])))
    vals.to_csv(out/'prediction_vs_reference.csv',index=False)
    vals.rename(columns={'X':'X','Y':'Y'}).to_csv(out/'grid_evaluation.csv',index=False)
    n=cfg.eval_grid_size
    yy,xx=np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n),indexing='ij')
    gridxy=np.column_stack([xx.ravel(),yy.ravel()]).astype(np.float32)
    gp=infer_macro(eq_net,gridxy,dev)
    grid=pd.DataFrame({'X':gridxy[:,0],'Y':gridxy[:,1],'rho':gp[:,0],
                       'u':gp[:,1],'v':gp[:,2],
                       'speed':np.hypot(gp[:,1],gp[:,2]),
                       'p_proxy_rho_over_3':gp[:,0]/3.0})
    grid.to_csv(out/'grid_data.csv',index=False)
    # Standard cavity centerline profiles: u(x=.5,y) and v(x,y=.5).
    nx=np.full((n,1),.5,np.float32)
    line=np.linspace(0,1,n,dtype=np.float32)[:,None]
    c1=np.hstack([nx,line]);c2=np.hstack([line,nx])
    po1=infer_macro(eq_net,c1,dev);po2=infer_macro(eq_net,c2,dev)
    pd.DataFrame(dict(y=line.ravel(),u_at_x_0p5=po1[:,1],
         x=line.ravel(),v_at_y_0p5=po2[:,2])).to_csv(out/'centerline_data.csv',index=False)
    msk=np.ones(len(df),dtype=bool);msk[ids]=False
    metric={'model':'KAN-SRT (dual KAN + BGK)','Re':1000,'true_nu_eval_only':cfg.true_nu,
            'nu_inferred':float(eq_net.nu_hat.detach().cpu().item()),
            'u_relative_L2_all':relative_l2(u,ref[:,0]),
            'v_relative_L2_all':relative_l2(v,ref[:,1]),
            'u_relative_L2_heldout':relative_l2(u[msk],ref[msk,0]) if msk.any() else None,
            'v_relative_L2_heldout':relative_l2(v[msk],ref[msk,1]) if msk.any() else None,
            'n_obs':len(ids),'n_heldout':int(msk.sum()),
            'parameters_eq':sum(p.numel() for p in eq_net.parameters()),
            'parameters_neq':sum(p.numel() for p in neq_net.parameters()),
            'parameters_total':sum(p.numel() for p in eq_net.parameters())+sum(p.numel() for p in neq_net.parameters()),
            'training_seconds':timing['total_optimizer_seconds'],
            'boundary_method':'SRT first-order CE using tau_hat=3*nu_hat+0.5'}
    if cfg.true_nu>0:
        metric['nu_relative_error_percent']=100*abs(metric['nu_inferred']-cfg.true_nu)/cfg.true_nu
    (out/'metrics.json').write_text(json.dumps(metric,indent=2,ensure_ascii=False),encoding='utf-8')
    pd.DataFrame([metric]).to_csv(out/'error_metrics.csv',index=False)
    torch.save({'eq_state_dict':eq_net.state_dict(),'neq_state_dict':neq_net.state_dict(),
                'config':asdict(cfg),'metrics':metric},out/'final_model.pt')
    fig,axs=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    refuv=[ref[:,0],u,np.abs(ref[:,0]-u),ref[:,1],v,np.abs(ref[:,1]-v)]
    titles=['u reference','u KAN-SRT','|u error|','v reference','v KAN-SRT','|v error|']
    for ax,z,title in zip(axs.ravel(),refuv,titles):
        im=ax.tricontourf(inp[:,0],inp[:,1],z,levels=40,cmap='jet')
        fig.colorbar(im,ax=ax);ax.set_title(title);ax.set_aspect('equal')
    fig.savefig(out/'velocity_error_maps.png',dpi=cfg.plot_dpi);plt.close(fig)
    if history:
        hh=pd.DataFrame(history)
        fig,ax=plt.subplots(figsize=(8,5))
        for key in ['total_loss','pde_loss','data_loss','boundary_loss']:
            ax.plot(hh['iteration'],hh[key],label=key,linewidth=1)
        ax.set_yscale('log');ax.set_xlabel('optimizer evaluations');ax.set_ylabel('Loss')
        ax.grid(alpha=.2);ax.legend();fig.tight_layout()
        fig.savefig(out/'loss_history.png',dpi=cfg.plot_dpi);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4))
        ax.plot(hh['iteration'],hh['nu_hat'],label='inferred nu')
        if cfg.true_nu>0:ax.axhline(cfg.true_nu,color='r',linestyle='--',label='eval reference')
        ax.grid(alpha=.3);ax.legend();ax.set_xlabel('optimizer evaluations');ax.set_ylabel('nu')
        fig.tight_layout();fig.savefig(out/'nu_history.png',dpi=cfg.plot_dpi);plt.close(fig)
    return metric


def run(cfg):
    if not (cfg.n_f>0 and cfg.n_b>0 and cfg.n_obs>0 and cfg.width>0 and cfg.epochs>=0):
        raise ValueError('n_f, n_b, n_obs, width must be positive; epochs nonnegative')
    datafile=Path(cfg.data)
    if not datafile.is_file():
        raise FileNotFoundError(f'Cannot read {datafile}; set --data path')
    seed_torch(cfg.seed)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df,ids=load_data(datafile,cfg.n_obs,cfg.data_seed)
    timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    out=Path(cfg.output) if cfg.output else SCRIPT_DIR/'results'/f'KAN_SRT_CE_Re1000_seed{cfg.seed}_{timestamp}'
    out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(asdict(cfg),ensure_ascii=False,indent=2),encoding='utf-8')
    initial_nu=(float(np.random.RandomState(cfg.seed).rand()*0.1)
                if cfg.initial_nu < 0 else cfg.initial_nu)
    eq_net=KAN_eq([2,cfg.width,3],grid_size=cfg.grid_size,spline_order=cfg.spline_order,
                  initial_nu_hat=initial_nu).to(dev)
    neq_net=KAN_neq([2,cfg.width,9],grid_size=cfg.grid_size,
                    spline_order=cfg.spline_order).to(dev)
    xi,w=d2q9(dev)
    xy_train=df.loc[ids,['X','Y']].to_numpy(np.float32)
    uv_train=df.loc[ids,['u','v']].to_numpy(np.float32)
    pd.DataFrame({'original_index':ids,'X':xy_train[:,0],'Y':xy_train[:,1],
                  'u':uv_train[:,0],'v':uv_train[:,1]}).to_csv(out/'training_observations.csv',index=False)
    xy_train=torch.as_tensor(xy_train,device=dev)
    uv_train=torch.as_tensor(uv_train,device=dev)
    rng=np.random.default_rng(cfg.seed)
    Xf=torch.as_tensor(rng.random((cfg.n_f,2)).astype('float32'),device=dev)
    pd.DataFrame(Xf.cpu().numpy(),columns=['X','Y']).to_csv(out/'physics_points.csv',index=False)
    bcs=make_bounds(cfg.n_b,rng,dev)
    for wall in bcs:
        a=wall['xy'].cpu().numpy();u=wall['uv'].cpu().numpy()
        pd.DataFrame({'X':a[:,0],'Y':a[:,1],'u_boundary':u[:,0],
                      'v_boundary':u[:,1]}).to_csv(out/f'boundary_{wall["name"]}.csv',index=False)
    # torch.Generator must be on matching device for randperm.
    gen=torch.Generator(device=dev).manual_seed(cfg.seed+121)
    opt=optim.Adam([{'params':[eq_net.nu_hat],'lr':cfg.lr_nu},
                    {'params':[p for k,p in eq_net.named_parameters() if k!='nu_hat'],'lr':cfg.lr},
                    {'params':neq_net.parameters(),'lr':cfg.lr}],lr=cfg.lr)
    history=[]
    elapsed_adam=0.;elapsed_lbfgs=0.;step_id=0
    def compute_losses():
        xbatch=draw_subset(Xf,cfg.batch_f,gen)
        pde=srt_pde_loss(xbatch,eq_net,neq_net,xi,w)
        obs=data_loss(eq_net,xy_train,uv_train)
        bc,details=boundary_loss(bcs,eq_net,neq_net,xi,w,cfg,gen)
        total=cfg.w_pde*pde+cfg.w_data*obs+cfg.w_bc*bc
        return total,pde,obs,bc,details
    def record(phase,step,values):
        total,pde,obs,bc,details=values
        ng=eq_net.nu_hat.grad
        history.append({'phase':phase,'iteration':step,'total_loss':float(total.detach()),
          'pde_loss':float(pde.detach()),'data_loss':float(obs.detach()),
          'boundary_loss':float(bc.detach()),'bc_macro':float(details[0].detach()),
          'bc_fneq':float(details[1].detach()),'bc_residual':float(details[2].detach()),
          'nu_hat':float(eq_net.nu_hat.detach().cpu()),
          'nu_grad':float(ng.detach().cpu()) if ng is not None else float('nan')})
    print(f'Using device={dev}; output={out}; model=dual-KAN SRT; boundary=CE with learned tau; nu_init={initial_nu:.7f}',flush=True)
    print(f'Settings: Adam={cfg.epochs}, LBFGS={cfg.lbfgs_max_iter}, '
          f'collocation={cfg.n_f}, boundary per wall={cfg.n_b}, obs={len(ids)}',flush=True)
    for epoch in range(1,cfg.epochs+1):
        tic=time.perf_counter()
        opt.zero_grad(set_to_none=True)
        vals=compute_losses();total=vals[0]
        if not torch.isfinite(total):raise FloatingPointError(f'Nonfinite loss at Adam step {epoch}')
        total.backward()
        torch.nn.utils.clip_grad_norm_(eq_net.parameters(),1.0)
        torch.nn.utils.clip_grad_norm_(neq_net.parameters(),1.0)
        opt.step()
        if dev.type=='cuda':torch.cuda.synchronize()
        elapsed_adam+=time.perf_counter()-tic
        record('Adam',epoch,vals)
        if epoch==1 or epoch%cfg.print_every==0:
            print(f'Adam {epoch}/{cfg.epochs}: total={float(total.detach()):.4e} '
                  f'pde={float(vals[1].detach()):.4e} data={float(vals[2].detach()):.4e} '
                  f'bc={float(vals[3].detach()):.4e} nu={float(eq_net.nu_hat.detach()):.7f}',flush=True)
        if cfg.save_every>0 and epoch%cfg.save_every==0:
            save_history(out,history)
            torch.save({'epoch':epoch,'eq':eq_net.state_dict(),'neq':neq_net.state_dict(),
                        'adam':opt.state_dict(),'config':asdict(cfg)},out/f'checkpoint_adam_{epoch}.pt')
    if cfg.lbfgs_max_iter>0:
        torch_opt=optim.LBFGS(list(eq_net.parameters())+list(neq_net.parameters()),lr=1e-2,
            max_iter=cfg.lbfgs_max_iter,max_eval=cfg.lbfgs_max_eval,
            tolerance_grad=1e-11,tolerance_change=1e-11,
            history_size=100,line_search_fn='strong_wolfe')
        counter=[0]
        # Maintain the same fixed samples during line-search closure evaluations.
        if cfg.batch_f>0 or cfg.batch_b>0:
            print('NOTICE: LBFGS line search uses full-batch losses for deterministic closure.',flush=True)
        oldbf,oldbb=cfg.batch_f,cfg.batch_b
        cfg.batch_f=cfg.batch_b=0
        def closure():
            torch_opt.zero_grad(set_to_none=True)
            vals=compute_losses()
            if not torch.isfinite(vals[0]):raise FloatingPointError('Nonfinite LBFGS loss')
            vals[0].backward()
            counter[0]+=1
            record('LBFGS',cfg.epochs+counter[0],vals)
            return vals[0]
        tic=time.perf_counter()
        torch_opt.step(closure)
        if dev.type=='cuda':torch.cuda.synchronize()
        elapsed_lbfgs=time.perf_counter()-tic
        cfg.batch_f,cfg.batch_b=oldbf,oldbb
    save_history(out,history)
    timing={'adam_seconds':elapsed_adam,'lbfgs_seconds':elapsed_lbfgs,
            'total_optimizer_seconds':elapsed_adam+elapsed_lbfgs,
            'adam_steps':cfg.epochs,'lbfgs_closure_calls':int(sum(e['phase']=='LBFGS' for e in history))}
    (out/'training_time.json').write_text(json.dumps(timing,indent=2),encoding='utf-8')
    metrics=save_final_results(cfg,df,ids,eq_net,neq_net,out,dev,history,timing)
    print('Final metrics:',json.dumps(metrics,indent=2),flush=True)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    from dataclasses import fields
    for f in fields(Config):
        default=f.default
        parser.add_argument('--'+f.name.replace('_','-'),type=type(default),default=default)
    args=parser.parse_args()
    run(Config(**{f.name:getattr(args,f.name) for f in fields(Config)}))

if __name__=='__main__':
    main()
