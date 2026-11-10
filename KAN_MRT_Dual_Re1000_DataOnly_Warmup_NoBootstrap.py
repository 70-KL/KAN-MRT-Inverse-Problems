#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re=1000 cavity: dual KAN (3+9 outputs) with collision-consistent kinetic boundaries.

This version emphasizes robust flow-field reconstruction and careful inverse
viscosity identification, NOT guaranteed recovery of the true viscosity.
The inferred quantity controls ONLY the two shear relaxation moments (7 and 8);
non-shear relaxation rates are fixed, independently configurable nuisance physics.

ARCHITECTURE IS UNCHANGED IN OUTPUT SIZE:
    KAN_eq  : (x,y) -> (rho,u,v)              [3 physical macro fields]
    KAN_neq : (x,y) -> (fneq_0,...,fneq_8)    [9 D2Q9 populations]

Corrections versus MRT-in-nu-Z-all-1000(3).py:
- Correct D2Q9 moment-index order, collision multiplication, and conserved rates.
- Use steady CONTINUOUS MRT (no discrete +0.5 relaxation shift).
- Low Mach equilibrium; map physical U=1 to kinetic velocity U_lattice=0.05.
- Direct 9-population neq OUTPUT is retained. Default differentiable moment
  projection removes its unphysical density/total momentum components.
- Inflow directions: e_i . n < 0; outflow e_i . n > 0; tangential omitted.
- Inflow defaults to corrected first-order Chapman-Enskog (CE)
  nonequilibrium fneq target; optional diffuse reflection can be selected.
  CE is an approximation, not an exact kinetic wall closure or discrete
  halfway-bounce-back MRT-LBM boundary.
- Outflow wall residual uses the SAME continuous collision as the interior.
- First stage is DATA ONLY (interior observed u/v): no macro wall, kinetic
  wall, or PDE losses are computed or used before physical ramp starts.
- The macro wall term is activated TOGETHER with PDE, CE inflow and MRT
  outgoing losses during the physical ramp, with the same ramp coefficient.
- Macro wall BC = (rho-1)^2+(u-u_wall)^2+(v-v_wall)^2, separate
  from DATA = observed u/v errors at observation sample points.
- Inverse viscosity: bounded damped analytic shear-profile update at frozen
  network fields (default), or alternating Adam if chosen. No target viscosity
  enters inverse training; an independent heldout flow grid is only evaluated.
- Optional auxiliary CE bootstrap is OFF by default (--w-ce 0). It can be
  enabled deliberately with --w-ce >0, without changing the CE WALL constraint.
- Both kinetic inflow and outgoing boundary constraints remain in the loss.
- Resuming restores random sampling state for reproducibility.

Note on units: if physical x in [0,1], Re = UL/nu_phys with U=L=1,
U_kin = alpha * U_phys, nu_kin = alpha * nu_phys,
s_nu = cs2/nu_kin, cs2=1/3. The equilibrium uses U_kin and the
continuous steady equation is e_i.grad f_i = - (M^-1 S M fneq)_i.

Requirements: torch, pandas, numpy, matplotlib, scipy.

Examples:
  python KAN_MRT_Dual_Re1000_DataOnly_Warmup_NoBootstrap.py --data num-1000.csv --mode forward
  python KAN_MRT_Dual_Re1000_DataOnly_Warmup_NoBootstrap.py --data num-1000.csv --mode inverse
  python KAN_MRT_Dual_Re1000_DataOnly_Warmup_NoBootstrap.py --data num-1000.csv \\
       --mode inverse --samples-from path/to/SINGLE_results_folder
  python KAN_MRT_Dual_Re1000_DataOnly_Warmup_NoBootstrap.py --data prediction_vs_reference.csv \\
       --epochs 2 --width 6 --grid-size 5 --batch-physics 4 --batch-wall 4 \\
       --batch-data 8 --profile-points 0 --output /tmp/mrt_quick

NOT a time-marching discrete MRT-LBM solver; no demonstrated full-run accuracy
until a complete training and numerical validation have been performed.
"""
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # ✅ 防止OpenMP库冲突
import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
E = [[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]]
W = [4/9] + [1/9]*4 + [1/36]*4
M = [
    [1,1,1,1,1,1,1,1,1],
    [-4,-1,-1,-1,-1,2,2,2,2],
    [4,-2,-2,-2,-2,1,1,1,1],
    [0,1,0,-1,0,1,-1,-1,1],
    [0,-2,0,2,0,1,-1,-1,1],
    [0,0,1,0,-1,1,1,-1,-1],
    [0,0,-2,0,2,1,1,-1,-1],
    [0,1,-1,1,-1,0,0,0,0],
    [0,0,0,0,0,1,-1,1,-1],
]
CS2 = 1 / 3
NONCONS = (1,2,4,6,7,8)
WALLS = ('top','left','right','bottom')
NORMALS = {'top':(0,1),'left':(-1,0),'right':(1,0),'bottom':(0,-1)}
WALL_UV = {'top':(1.,0.),'left':(0.,0.),'right':(0.,0.),'bottom':(0.,0.)}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class KANLinear(nn.Module):
    """Original SiLU + cubic B-spline KANLinear, with corrected [-1,1] grid."""
    def __init__(self, ni, no, grid_size=15, order=3):
        super().__init__()
        self.ni, self.no, self.grid_size, self.order = ni, no, grid_size, order
        h = 2. / grid_size
        grid = (torch.arange(-order, grid_size + order + 1)*h-1.).expand(ni,-1).contiguous()
        self.register_buffer('grid', grid)
        self.base_weight = nn.Parameter(torch.empty(no,ni))
        self.spline_weight = nn.Parameter(torch.empty(no,ni,grid_size+order))
        self.spline_scaler = nn.Parameter(torch.empty(no,ni))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5))
        with torch.no_grad():
            knots = self.grid.T[order:-order]
            noise = ((torch.rand(grid_size+1, ni, no)-.5)*.1/grid_size)
            self.spline_weight.copy_(torch.linalg.lstsq(self.bsplines(knots).transpose(0,1),
                                                        noise.transpose(0,1)).solution.permute(2,0,1))

    def bsplines(self,x):
        xx=x.unsqueeze(-1)
        g=self.grid
        b=((xx>=g[:,:-1])&(xx<g[:,1:])).to(x.dtype)
        for k in range(1,self.order+1):
            b=((xx-g[:,:-(k+1)])/(g[:,k:-1]-g[:,:-(k+1)])*b[:,:,:-1]
               +(g[:,k+1:]-xx)/(g[:,k+1:]-g[:,1:-k])*b[:,:,1:])
        return b.contiguous()

    def forward(self,x):
        spline=self.spline_weight*self.spline_scaler.unsqueeze(-1)
        base=F.linear(F.silu(x),self.base_weight)
        return base+F.linear(self.bsplines(x).reshape(x.shape[0],-1),spline.reshape(self.no,-1))


class KAN(nn.Module):
    def __init__(self, nout, width, depth, grid_size):
        super().__init__()
        dims=[2]+[width]*depth+[nout]
        self.layers=nn.ModuleList([KANLinear(a,b,grid_size) for a,b in zip(dims[:-1],dims[1:])])

    def forward(self,xy):
        z=2.*xy-1.
        for j,layer in enumerate(self.layers):
            z=layer(z)
            if j < len(self.layers)-1:
                z=torch.tanh(z)
        return z


class MacroKAN(KAN):
    """KAN_eq(x,y) returns the physical (rho,u,v), exactly three channels."""
    def __init__(self,width,depth,grid_size,kinetic_u):
        super().__init__(3,width,depth,grid_size)
        self.kinetic_u=float(kinetic_u)

    def forward(self,xy):
        raw=super().forward(xy)
        # Fixed low-Mach output normalization, not a separate learned p network.
        return torch.cat([1.+self.kinetic_u**2*raw[:,:1],raw[:,1:3]],dim=1)


class NonEquilibriumKAN(KAN):
    """KAN_neq(x,y) returns exactly nine physical fneq populations."""
    def __init__(self,width,depth,grid_size,neq_scale,project_neq,mt,mi,mask):
        super().__init__(9,width,depth,grid_size)
        self.neq_scale=float(neq_scale)
        self.project_neq=bool(project_neq)
        self.register_buffer('moment_matrix',mt.clone())
        self.register_buffer('inverse_moment_matrix',mi.clone())
        self.register_buffer('noncons_mask',mask.clone())

    def forward(self,xy):
        raw=super().forward(xy)*self.neq_scale
        if self.project_neq:
            raw=(raw@self.moment_matrix.T*self.noncons_mask)@self.inverse_moment_matrix.T
        return raw


class DualKANMRT(nn.Module):
    """Explicit KAN_eq:3 outputs and KAN_neq:9 outputs."""
    def __init__(self,cfg):
        super().__init__()
        self.log_nu=nn.Parameter(torch.tensor(math.log(cfg.initial_nu),dtype=torch.float64 if cfg.dtype=='float64' else torch.float32))
        self.kinetic_u=float(cfg.kinetic_u)
        self.neq_scale=float(self.kinetic_u**2*cfg.neq_scale_nu/CS2)
        self.project_neq=bool(cfg.project_neq)
        # User-configurable fixed NON-SHEAR continuous collision rates.
        self.nonshear_e=float(cfg.nonshear_e)
        self.nonshear_eps=float(cfg.nonshear_eps)
        self.nonshear_q=float(cfg.nonshear_q)
        self.register_buffer('e',torch.tensor(E,dtype=self.log_nu.dtype))
        self.register_buffer('w',torch.tensor(W,dtype=self.log_nu.dtype))
        mt=torch.tensor(M,dtype=self.log_nu.dtype)
        self.register_buffer('M',mt)
        self.register_buffer('Minv',torch.linalg.inv(mt))
        self.register_buffer('noncons_mask',torch.tensor([0,1,1,0,1,0,1,1,1],dtype=self.log_nu.dtype))
        self.macro_net=MacroKAN(cfg.width,cfg.depth,cfg.grid_size,self.kinetic_u)
        self.neq_net=NonEquilibriumKAN(cfg.width,cfg.depth,cfg.grid_size,
                                       self.neq_scale,self.project_neq,self.M,
                                       self.Minv,self.noncons_mask)

    @property
    def nu(self): return self.log_nu.exp()

    def collision_rates(self,nu_override=None):
        nu=self.nu if nu_override is None else nu_override
        s_shear=CS2/(self.kinetic_u*nu)
        # 0:rho 1:e 2:eps 3:jx 4:qx 5:jy 6:qy 7:pxx 8:pxy
        # The known shear viscosity changes ONLY modes pxx (7), pxy (8).
        # e, eps, qx, qy relax independently at fixed positive rates; they are
        # NOT hidden additional viscosity-dependent parameters of the inverse.
        rates=torch.stack([
            torch.zeros_like(s_shear),
            torch.ones_like(s_shear)*self.nonshear_e,
            torch.ones_like(s_shear)*self.nonshear_eps,
            torch.zeros_like(s_shear),
            torch.ones_like(s_shear)*self.nonshear_q,
            torch.zeros_like(s_shear),
            torch.ones_like(s_shear)*self.nonshear_q,
            s_shear,s_shear
        ])
        return rates

    def macro(self,xy):
        macro=self.macro_net(xy)
        return macro[:,:1],macro[:,1:2],macro[:,2:3]

    def neq(self,xy):
        return self.neq_net(xy)

    def equilibrium(self,rho,u,v):
        uv=self.kinetic_u*torch.cat([u,v],dim=1)
        eu=uv@self.e.T
        speed2=uv.square().sum(dim=1,keepdim=True)
        return rho*self.w*(1.+eu/CS2+eu.square()/(2*CS2**2)-speed2/(2*CS2))

    def populations(self,xy):
        rho,u,v=self.macro(xy)
        fneq=self.neq(xy)
        feq=self.equilibrium(rho,u,v)
        return feq+fneq,feq,fneq,rho,u,v


def grad(scalar,xy,create_graph=True):
    return torch.autograd.grad(scalar,xy,torch.ones_like(scalar),
                               retain_graph=True,create_graph=create_graph)[0]


def advect(f,xy,e,create_graph=True):
    df=torch.stack([grad(f[:,i:i+1],xy,create_graph) for i in range(9)],dim=1)
    return (df*e.unsqueeze(0)).sum(dim=2)


def kinetic_residual(model,xy,nu_override=None,create_graph=True):
    f,feq,fneq,rho,u,v=model.populations(xy)
    advection=advect(f,xy,model.e,create_graph)
    mneq=fneq@model.M.T
    rate=model.collision_rates(nu_override)
    r_m=advection@model.M.T+mneq*rate
    # row convention r_f = (df/dt + e.grad f) + fneq @ C.T
    r_f=r_m@model.Minv.T
    return r_m,r_f,mneq,rho,u,v,f,feq,fneq


def interior_losses(model,xy,cfg):
    r_m,_,mneq,rho,u,v,_,_,_=kinetic_residual(model,xy)
    eps=model.kinetic_u
    mass=(r_m[:,0]/eps).square().mean()
    momentum=((r_m[:,[3,5]]/eps**2)**2).mean()
    noncons=(r_m[:,list(NONCONS)]/eps).square().mean()
    div=(grad(u,xy)[:,0]+grad(v,xy)[:,1]).square().mean()
    rho_reg=((rho-1.)/eps**2).square().mean()
    moment=((mneq[:,[0,3,5]]/model.neq_scale)**2).mean()
    combined=(cfg.w_mass*mass+cfg.w_momentum*momentum+cfg.w_kinetic*noncons
              +cfg.w_div*div+cfg.w_rho*rho_reg
              +(0. if cfg.project_neq else cfg.w_moment*moment))
    return {'physics':combined,'mass':mass,'momentum':momentum,'kinetic':noncons,
            'div':div,'rho_reg':rho_reg,'moment':moment}


def macro_wall_terms(rho,u,v,name):
    """Only FOUR WALL samples: density rho=1 and prescribed wall u,v.

    Unlike observation/data loss, this is a boundary-condition loss. The
    reference velocity labels and interior observation coordinates are never
    used here. Returns separately loggable density and velocity penalties.
    """
    ub,vb=WALL_UV[name]
    loss_rho=(rho-1.0).square().mean()
    loss_uv=((u-ub).square()+(v-vb).square()).mean()
    return loss_rho,loss_uv


def wall_loss(model,xy,name,cfg):
    """Kinetic inflow e.n<0; kinetic outflow e.n>0, excluding tangents."""
    n=torch.as_tensor(NORMALS[name],dtype=xy.dtype,device=xy.device)
    goal=torch.as_tensor(WALL_UV[name],dtype=xy.dtype,device=xy.device)
    f,feq,fneq,rho,u,v=model.populations(xy)
    loss_rho,loss_uv=macro_wall_terms(rho,u,v,name)
    macro=loss_rho+loss_uv
    dot=model.e@n
    i_in=torch.nonzero(dot<-.5,as_tuple=True)[0]
    i_out=torch.nonzero(dot>.5,as_tuple=True)[0]

    if cfg.inflow_model=='diffuse':
        # Diffuse reflection with local wall density determined by zero normal
        # mass flux; wall equilibrium uses prescribed wall tangential velocity.
        wall_uv=goal.unsqueeze(0).expand(len(xy),-1)
        wall_eu=(model.kinetic_u*wall_uv)@model.e.T
        wall_v2=(model.kinetic_u*wall_uv).square().sum(dim=1,keepdim=True)
        wall_eq_unit=model.w*(1+wall_eu/CS2+wall_eu.square()/(2*CS2**2)-wall_v2/(2*CS2))
        out_flux=(f[:,i_out]*dot[i_out]).sum(dim=1,keepdim=True)
        in_flux=(wall_eq_unit[:,i_in]*dot[i_in]).sum(dim=1,keepdim=True)
        wall_density=torch.clamp(-out_flux/in_flux,min=.2,max=3.)
        incoming_target=wall_density*wall_eq_unit[:,i_in]
        loss_in=((f[:,i_in]-incoming_target)/model.kinetic_u).square().mean()
    elif cfg.inflow_model=='ce':
        # First-order Chapman--Enskog approx, *not* an exact wall BC.
        dfeq=advect(feq,xy,model.e)
        sm=model.collision_rates()
        inverse_sm=torch.where(sm>0,1./torch.clamp(sm,min=1e-30),torch.zeros_like(sm))
        m_star=-(dfeq@model.M.T)*inverse_sm
        fneq_star=m_star@model.Minv.T
        loss_in=((fneq[:,i_in]-fneq_star[:,i_in])/model.neq_scale).square().mean()
    else:
        raise ValueError(cfg.inflow_model)

    # On the outgoing directions the exact same PDE is enforced as interior.
    advection=advect(f,xy,model.e)
    mneq=fneq@model.M.T
    r_f=(advection@model.M.T + mneq*model.collision_rates())@model.Minv.T
    loss_out=(r_f[:,i_out]/model.kinetic_u).square().mean()
    return {'mbc':macro,'mbc_rho':loss_rho,'mbc_uv':loss_uv,'fbc':loss_in,'Ebc':loss_out}


@dataclass
class Config:
    data: str = ''
    output: str = ''
    samples_from: str = ''
    mode: str = 'inverse'   # 'inverse' or 'forward'
    epochs: int = 40000
    warmup_epochs: int = 2000
    physics_ramp_epochs: int = 5000
    n_train: int = 1000
    n_f: int = 20000
    n_b: int = 5000
    corner_exclusion: float = .005  # avoid incompatible top-lid corner velocities
    batch_data: int = 256
    batch_physics: int = 128
    batch_wall: int = 48
    refine_top_fraction: float = .25  # sample more existing pool points near lid
    refine_corner_fraction: float = .02  # keep modest: corners are singular
    width: int = 30
    depth: int = 1
    grid_size: int = 15
    dtype: str = 'float64'  # preserve tiny fneq + derivative precision
    seed: int = 42
    initial_nu: float = .01
    true_nu: float = .001  # forward fixed; inverse evaluation only
    neq_scale_nu: float = .003  # fixed output scale; NOT a viscosity prior/target
    kinetic_u: float = .05
    project_neq: int = 1
    inflow_model: str = 'ce'  # default: corrected CE fneq; optional 'diffuse'
    w_data: float = 3.
    w_mbc: float = 3.
    w_fbc: float = .1
    w_Ebc: float = .1
    w_mass: float = 1.
    w_momentum: float = 1.
    w_kinetic: float = 1.
    w_div: float = .1
    w_rho: float = .001
    w_ce: float = 0.0  # DEFAULT OFF: optional CE interior bootstrap, NOT CE wall BC
    w_moment: float = 1.
    lr: float = .001
    nu_lr: float = .0001
    nu_min: float = 1e-5
    nu_max: float = .1
    net_steps: int = 5
    nu_steps: int = 1
    nu_update: str = 'analytic'  # analytic (recommended) or adam
    nu_damping: float = .25   # log-space fraction toward accepted candidate
    nu_max_log_step: float = .12  # bound change per update
    nu_batch: int = 384
    nu_validation_batch: int = 192
    nu_min_shear_signal: float = 1e-17
    nonshear_e: float = 800.0
    nonshear_eps: float = 800.0
    nonshear_q: float = 1100.0
    decay_step: int = 5000
    decay_gamma: float = .93
    print_every: int = 200
    save_every: int = 5000
    eval_batch: int = 2048
    profile_points: int = 31
    profile_batch: int = 256
    device: str = 'auto'
    resume: str = ''


def load_data(filename):
    df=pd.read_csv(filename)
    df.columns=df.columns.str.strip()
    if all(q in df.columns for q in ('X','Y','u','v')):
        return df[['X','Y','u','v']].to_numpy(np.float64)
    if all(q in df.columns for q in ('X','Y','u_reference','v_reference')):
        return df[['X','Y','u_reference','v_reference']].to_numpy(np.float64)
    raise ValueError('CSV needs X,Y,u,v or X,Y,u_reference,v_reference')


def pool_wall(name,n,rng,margin):
    s=rng.uniform(margin,1-margin,(n,1))
    z=np.zeros_like(s);o=np.ones_like(s)
    if name=='top': return np.concatenate([s,o],axis=1)
    if name=='bottom': return np.concatenate([s,z],axis=1)
    if name=='left': return np.concatenate([z,s],axis=1)
    if name=='right': return np.concatenate([o,s],axis=1)
    raise ValueError(name)


def load_pools(cfg,values):
    rng=np.random.default_rng(cfg.seed)
    src=Path(cfg.samples_from) if cfg.samples_from else None
    if src:
        obs=pd.read_csv(src/'training_observations.csv')
        obs.columns=obs.columns.str.strip()
        if not set(['X','Y']).issubset(obs): raise ValueError('training_observations.csv needs X,Y')
        xy=obs[['X','Y']].to_numpy(np.float64)
        # Original single-file labels have u,v; NS variant has u_train,v_train.
        if {'u','v'}.issubset(obs.columns): y=obs[['u','v']].to_numpy(np.float64)
        elif {'u_train','v_train'}.issubset(obs.columns): y=obs[['u_train','v_train']].to_numpy(np.float64)
        else:
            indices=cKDTree(values[:,:2]).query(xy)[1]
            y=values[indices,2:4]
        f_pool=pd.read_csv(src/'physics_points.csv')[['X','Y']].to_numpy(np.float64)
        walls={k:pd.read_csv(src/f'boundary_{k}.csv')[['X','Y']].to_numpy(np.float64)
               for k in WALLS}
    else:
        n=min(cfg.n_train,len(values))
        ids=np.sort(rng.choice(len(values),n,replace=False))
        xy=values[ids,:2]
        y=values[ids,2:4]
        f_pool=rng.random((cfg.n_f,2))
        walls={k:pool_wall(k,cfg.n_b,rng,cfg.corner_exclusion) for k in WALLS}
    tree=cKDTree(values[:,:2]); dist,indices=tree.query(xy)
    if float(np.max(dist))>2e-5: raise ValueError('Observed coordinate absent from supplied reference CSV')
    is_train=np.zeros(len(values),bool);is_train[indices]=True
    return xy,y,f_pool,walls,is_train


def take(pool,n,rng,device,dtype,needs_grad=False):
    ids=rng.integers(len(pool),size=n)
    x=torch.as_tensor(pool[ids],device=device,dtype=dtype)
    return x.detach().requires_grad_(needs_grad)


def take_physics(pool,n,rng,device,dtype,cfg):
    n_top=int(round(n*cfg.refine_top_fraction))
    n_corner=int(round(n*cfg.refine_corner_fraction))
    n_global=max(0,n-n_top-n_corner)
    idx_top=np.flatnonzero(pool[:,1]>.8)
    idx_corner=np.flatnonzero((pool[:,0]>.8)&(pool[:,1]>.8))
    if not len(idx_top):n_global+=n_top;n_top=0
    if not len(idx_corner):n_global+=n_corner;n_corner=0
    choices=[rng.integers(len(pool),size=n_global)]
    if n_top:choices.append(rng.choice(idx_top,size=n_top,replace=True))
    if n_corner:choices.append(rng.choice(idx_corner,size=n_corner,replace=True))
    indices=np.concatenate(choices)
    rng.shuffle(indices)
    return torch.as_tensor(pool[indices],device=device,dtype=dtype).detach().requires_grad_(True)


def metric(pred,ref):
    return {'L2_u':float(np.linalg.norm(pred[:,0]-ref[:,0])/max(np.linalg.norm(ref[:,0]),1e-12)),
            'L2_v':float(np.linalg.norm(pred[:,1]-ref[:,1])/max(np.linalg.norm(ref[:,1]),1e-12)),
            'L2_uv':float(np.linalg.norm(pred-ref)/max(np.linalg.norm(ref),1e-12)),
            'MAE_u':float(np.mean(np.abs(pred[:,0]-ref[:,0]))),
            'MAE_v':float(np.mean(np.abs(pred[:,1]-ref[:,1])))}


def evaluate(model,values,mask,cfg,device,dtype,out):
    model.eval()
    xy=values[:,:2]
    preds=[]
    with torch.no_grad():
        for i in range(0,len(xy),cfg.eval_batch):
            x=torch.as_tensor(xy[i:i+cfg.eval_batch],device=device,dtype=dtype)
            rho,u,v=model.macro(x)
            fneq=model.neq(x)
            preds.append(torch.cat([rho,u,v,fneq],dim=1).cpu().numpy())
    pp=np.concatenate(preds,axis=0)
    uv=pp[:,1:3]; ref=values[:,2:4]
    frame=pd.DataFrame({'X':xy[:,0],'Y':xy[:,1],
                        'u_reference':ref[:,0], 'v_reference':ref[:,1],
                        'u_pred':uv[:,0],'v_pred':uv[:,1],
                        'u_abs_error':np.abs(uv[:,0]-ref[:,0]),
                        'v_abs_error':np.abs(uv[:,1]-ref[:,1]),
                        'rho_pred':pp[:,0], 'is_training_point':mask.astype(int),
                        **{f'fneq_{i}':pp[:,3+i] for i in range(9)}})
    frame.to_csv(out/'prediction_vs_reference.csv',index=False)
    grid=frame[['X','Y','u_pred','v_pred','rho_pred']].rename(columns={'u_pred':'u','v_pred':'v','rho_pred':'rho'})
    grid.to_csv(out/'grid_data.csv',index=False)
    stats={'full_grid':metric(uv,ref),'training_subset':metric(uv[mask],ref[mask]),
           'heldout_only':metric(uv[~mask],ref[~mask]) if (~mask).any() else None,
           'nu_inferred':float(model.nu.detach().cpu()),
           'nu_reference_post_training':cfg.true_nu,
           'nu_relative_error_percent':100*abs(float(model.nu.detach().cpu())-cfg.true_nu)/cfg.true_nu,
           'network_1_output':'rho,u,v','network_2_output':'fneq_0...fneq_8',
           'total_parameters':sum(p.numel() for p in model.parameters())}
    (out/'metrics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame([{'partition':'full_grid',**stats['full_grid']},
                  {'partition':'training_subset',**stats['training_subset']},
                  *([{'partition':'heldout_only',**stats['heldout_only']}] if stats['heldout_only'] else [])]).to_csv(
                      out/'error_metrics.csv',index=False)
    for title,col in [('u absolute error','u_abs_error'),('v absolute error','v_abs_error')]:
        fig,ax=plt.subplots(figsize=(5.5,4.7))
        z=ax.tricontourf(frame.X,frame.Y,frame[col],levels=50)
        fig.colorbar(z,ax=ax)
        ax.set(xlabel='x',ylabel='y',title=title,aspect='equal')
        fig.tight_layout();fig.savefig(out/(col+'.png'),dpi=180);plt.close(fig)
    # Physically correct centerlines: u(x,0.5) and v(0.5,y)
    line=np.linspace(0,1,201)
    q1=np.column_stack([line,np.full_like(line,.5)])
    q2=np.column_stack([np.full_like(line,.5),line])
    with torch.no_grad():
        u_mid=model.macro(torch.as_tensor(q1,device=device,dtype=dtype))[1].cpu().numpy().ravel()
        v_mid=model.macro(torch.as_tensor(q2,device=device,dtype=dtype))[2].cpu().numpy().ravel()
    ur=griddata(xy,ref[:,0],q1,method='linear')
    vr=griddata(xy,ref[:,1],q2,method='linear')
    pd.DataFrame({'coordinate':line,'u_at_y0p5_pred':u_mid,'u_at_y0p5_ref':ur,
                  'v_at_x0p5_pred':v_mid,'v_at_x0p5_ref':vr}).to_csv(out/'centerline_profiles.csv',index=False)
    return stats


def profile(model,cfg,physics_pool,device,dtype,out):
    if cfg.profile_points<2:return
    # Frozen-network check: fixed field & derivatives, only s_nu(nu) varies.
    xy=torch.as_tensor(physics_pool[:max(2,cfg.profile_batch)],dtype=dtype,device=device).requires_grad_(True)
    _,_,mneq,_,_,_,f,_,_=kinetic_residual(model,xy)
    D=advect(f,xy,model.e).detach()@model.M.T
    m=mneq.detach()
    eps=model.kinetic_u
    arr=[]
    for nu in np.geomspace(cfg.nu_min,cfg.nu_max,cfg.profile_points):
        rates=model.collision_rates(torch.tensor(nu,dtype=dtype,device=device)).detach()
        R=D+m*rates
        score=(R[:,[0]]/eps).square().mean()+(R[:,[3,5]]/eps**2).square().mean()+(R[:,list(NONCONS)]/eps).square().mean()
        arr.append({'nu':nu,'physics_score':float(score.item())})
    pd.DataFrame(arr).to_csv(out/'nu_profile.csv',index=False)
    fig,ax=plt.subplots(figsize=(6,4))
    ax.loglog([r['nu'] for r in arr],[r['physics_score'] for r in arr])
    ax.axvline(float(model.nu.detach()),linestyle='--',label='trained nu')
    ax.set(xlabel='test viscosity (frozen network)',ylabel='moment residual score')
    ax.legend();fig.tight_layout();fig.savefig(out/'nu_profile.png',dpi=160);plt.close(fig)


def parse():
    ap=argparse.ArgumentParser(description='Corrected dual KAN (rho,u,v)+(9 fneq) continuous MRT, with in/out wall losses')
    for field in Config.__dataclass_fields__.values():
        k=field.name
        val=getattr(Config,k)
        key='--'+k.replace('_','-')
        if isinstance(val,int): ap.add_argument(key,type=int,default=val)
        elif isinstance(val,float):ap.add_argument(key,type=float,default=val)
        else:ap.add_argument(key,type=str,default=val)
    cfg=Config(**vars(ap.parse_args()))
    if cfg.mode not in ('forward','inverse') or cfg.inflow_model not in ('diffuse','ce') or cfg.nu_update not in ('analytic','adam'):
        ap.error('mode: forward/inverse; inflow-model: diffuse/ce; nu-update: analytic/adam')
    if cfg.dtype not in ('float32','float64') or cfg.epochs<1 or cfg.batch_physics<1 or cfg.batch_wall<1:
        ap.error('dtype float32/float64; epochs, batch sizes > 0')
    if cfg.refine_top_fraction<0 or cfg.refine_corner_fraction<0 or cfg.refine_top_fraction+cfg.refine_corner_fraction>1:
        ap.error('refine-top-fraction and refine-corner-fraction must be nonnegative and sum <= 1')
    if cfg.initial_nu<=0 or cfg.true_nu<=0 or cfg.kinetic_u<=0 or cfg.neq_scale_nu<=0 or min(cfg.nonshear_e,cfg.nonshear_eps,cfg.nonshear_q)<=0:
        ap.error('positive nu/kinetic-u/neq-scale-nu/nonshear rates required')
    if cfg.nu_damping<=0 or cfg.nu_damping>1 or cfg.nu_batch<2 or cfg.nu_validation_batch<2 or cfg.nu_max_log_step<=0:
        ap.error('nu-damping in (0,1], nu batches >=2 and nu-max-log-step>0 required')
    if cfg.nu_min>=cfg.nu_max or cfg.nu_min<=0 or cfg.net_steps<1 or cfg.nu_steps<1:
        ap.error('nu bounds must be positive/increasing; net-steps and nu-steps >=1')
    if not 0<cfg.corner_exclusion<.25:
        ap.error('corner-exclusion must be between 0 and .25')
    return cfg


def ce_bootstrap_loss(model,xy):
    """Soft one-term CE anchor from the current macro equilibrium; NOT ground truth.

    First-order quasi-steady: S * mneq ~= -M (e.grad feq).
    Its accuracy is limited at walls/strong gradients; switch off after ramp.
    """
    rho,u,v=model.macro(xy)
    feq=model.equilibrium(rho,u,v)
    dfeq=advect(feq,xy,model.e)
    rates=model.collision_rates()
    inv=torch.where(rates>0,1.0/torch.clamp(rates,min=1e-30),torch.zeros_like(rates))
    m_ce=-(dfeq@model.M.T)*inv
    m_pred=model.neq(xy)@model.M.T
    return ((m_pred[:,list(NONCONS)]-m_ce[:,list(NONCONS)])/model.neq_scale).square().mean()


def shear_profile_components(model,xy):
    """Frozen field A+m*s in the two shear moments; no network parameter grad."""
    with torch.enable_grad():
        x=xy.detach().requires_grad_(True)
        f,feq,fneq,_,_,_=model.populations(x)
        D=advect(f,x,model.e,create_graph=False).detach()@model.M.T
        m=(fneq.detach()@model.M.T)[:,[7,8]]
        a=D[:,[7,8]]
    return a.detach(),m.detach()


def shear_score(a,m,s):
    return float((a+s*m).square().mean().item())


def analytic_nu_update(model,pool,rng,cfg,device,dtype):
    """Minimize the two shear residuals exactly in collision frequency s>=0,
    then damp, trust-region clip, and accept only if a heldout physics mini-batch
    does not worsen (network frozen throughout).

    This solves a one-dimensional *conditional* minimization, not the global
    inverse problem. Insufficient stress signal returns a skipped update.
    """
    train_x=take(pool,cfg.nu_batch,rng,device,dtype,needs_grad=True)
    val_x=take(pool,cfg.nu_validation_batch,rng,device,dtype,needs_grad=True)
    a,m=shear_profile_components(model,train_x)
    den=float(m.square().mean().item())
    current_nu=float(model.nu.detach().item())
    before_s=CS2/(model.kinetic_u*current_nu)
    rec={'nu_candidate':float('nan'),'shear_signal':den,'shear_score_old':float('nan'),
         'shear_score_new':float('nan'),'nu_accepted':0}
    if not math.isfinite(den) or den<cfg.nu_min_shear_signal:
        return rec
    numerator=float((-a*m).mean().item())
    cand_s=numerator/den
    if not math.isfinite(cand_s) or cand_s<=0:
        return rec
    candidate=min(cfg.nu_max,max(cfg.nu_min,CS2/(model.kinetic_u*cand_s)))
    rec['nu_candidate']=candidate
    old_log=math.log(current_nu)
    proposed_log=old_log+np.clip(cfg.nu_damping*(math.log(candidate)-old_log),
                                 -cfg.nu_max_log_step,cfg.nu_max_log_step)
    proposal=min(cfg.nu_max,max(cfg.nu_min,math.exp(float(proposed_log))))
    a_val,m_val=shear_profile_components(model,val_x)
    old_score=shear_score(a_val,m_val,before_s)
    new_score=shear_score(a_val,m_val,CS2/(model.kinetic_u*proposal))
    rec['shear_score_old']=old_score;rec['shear_score_new']=new_score
    if not math.isfinite(new_score) or new_score>old_score*(1+1e-3):
        return rec
    with torch.no_grad():
        model.log_nu.fill_(math.log(proposal))
    rec['nu_accepted']=1
    return rec


def main(cfg):
    seed_all(cfg.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if cfg.device=='auto' else torch.device(cfg.device)
    dtype=torch.float64 if cfg.dtype=='float64' else torch.float32
    data=cfg.data or str(HERE/'num-1000.csv')
    if not Path(data).exists():
        alt=HERE/'prediction_vs_reference.csv'
        if not cfg.data and alt.exists():data=str(alt)
        else:raise FileNotFoundError(data)
    values=load_data(data)
    if len(values)<2 or not np.isfinite(values).all():
        raise ValueError('Reference data needs >=2 rows and all values must be finite')
    if np.any(values[:,:2]<-1e-6) or np.any(values[:,:2]>1.+1e-6):
        raise ValueError('This Re1000 cavity implementation expects X,Y in [0,1]')
    obs,targets,pool,walls,mask=load_pools(cfg,values)
    print(f'Reference dataset: {Path(data).resolve()} (rows={len(values)})',flush=True)
    if not cfg.data and Path(data).name=='prediction_vs_reference.csv':
        print('INFO: num-1000.csv was not found; using u_reference,v_reference from prediction_vs_reference.csv, NOT u_pred/v_pred.',flush=True)
    if cfg.mode=='inverse':
        start_update=cfg.warmup_epochs+cfg.physics_ramp_epochs+cfg.net_steps+1
        if cfg.epochs<start_update:
            print(f'WARNING: no viscosity update occurs until approximately epoch {start_update}; with --epochs {cfg.epochs}, nu will remain at the initial value.',flush=True)
        else:
            print(f'Inverse schedule: fixed nu in the initial {cfg.warmup_epochs+cfg.physics_ramp_epochs} epochs; first conditional nu update near epoch {start_update}.',flush=True)
    if device.type=='cpu' and cfg.epochs>=1000:
        print('NOTE: running full high-precision MRT training on CPU can be very compute-intensive; CUDA GPU is recommended.',flush=True)
    out=Path(cfg.output) if cfg.output else (HERE/'results_dual_KAN_MRT_optimized'/
                                          (cfg.mode+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(asdict(cfg),ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame({'X':obs[:,0],'Y':obs[:,1],'u':targets[:,0],'v':targets[:,1]}).to_csv(out/'training_observations.csv',index=False)
    pd.DataFrame(pool,columns=['X','Y']).to_csv(out/'physics_points.csv',index=False)
    for name in WALLS:pd.DataFrame(walls[name],columns=['X','Y']).to_csv(out/f'boundary_{name}.csv',index=False)
    pd.DataFrame(values[~mask],columns=['X','Y','u','v']).to_csv(out/'heldout_observations.csv',index=False)

    model=DualKANMRT(cfg).to(device=device,dtype=dtype)
    if cfg.mode=='forward':
        with torch.no_grad():model.log_nu.fill_(math.log(cfg.true_nu))
    network_params=list(model.macro_net.parameters())+list(model.neq_net.parameters())
    optim_net=torch.optim.Adam(network_params,lr=cfg.lr)
    optim_nu=torch.optim.Adam([model.log_nu],lr=cfg.nu_lr)
    scheduler_net=torch.optim.lr_scheduler.StepLR(optim_net,step_size=max(1,cfg.decay_step),gamma=cfg.decay_gamma)
    scheduler_nu=torch.optim.lr_scheduler.StepLR(optim_nu,step_size=max(1,cfg.decay_step),gamma=cfg.decay_gamma)
    start_epoch=0
    hist=[]
    if cfg.resume:
        ckpt=torch.load(cfg.resume,map_location=device,weights_only=False)
        if ckpt.get('config',{}).get('mode')!=cfg.mode:
            raise ValueError('Cannot resume from checkpoint of different forward/inverse mode')
        model.load_state_dict(ckpt['model']);optim_net.load_state_dict(ckpt['net_opt']);optim_nu.load_state_dict(ckpt['nu_opt']);
        scheduler_net.load_state_dict(ckpt['net_sched']);scheduler_nu.load_state_dict(ckpt['nu_sched'])
        start_epoch=ckpt['epoch'];hist=ckpt['history']
        # When resuming, --samples-from <old_results> reproduces exact sampling pools.
        if not cfg.samples_from:raise ValueError('Resuming requires --samples-from ORIGINAL_RESULT_DIR')
    print(f'device={device} dtype={dtype} mode={cfg.mode} epochs={cfg.epochs} '
          f'N_obs={len(obs)} N_f={len(pool)} N_b={len(walls["top"])} each '
          f'KAN=(2,{cfg.width},3)/(2,{cfg.width},9), nu_initial={float(model.nu.detach()):.7g}, '
          f'Mach={math.sqrt(3)*cfg.kinetic_u:.4g}, in_wall={cfg.inflow_model}, nu_update={cfg.nu_update}, '
          f'CE_bootstrap={"OFF" if cfg.w_ce==0 else cfg.w_ce}; '
          f'warmup=DATA_ONLY({cfg.warmup_epochs} epochs)',flush=True)
    print('Output directory:',out,flush=True)
    rng=np.random.default_rng(cfg.seed+111)
    if cfg.resume and 'rng_state' in ckpt:
        rng.bit_generator.state=ckpt['rng_state']
    history_start=time.perf_counter()
    for epoch in range(start_epoch+1,cfg.epochs+1):
        # Stage 1: data-only. Stage 2: simultaneously activate PDE + ALL
        # three wall losses (macro, CE incoming, MRT outgoing) with alpha.
        # No wall samples/physics graph are computed during data-only stage.
        if epoch<=cfg.warmup_epochs: alpha=0.
        else: alpha=min(1.,(epoch-cfg.warmup_epochs)/max(1,cfg.physics_ramp_epochs))
        if cfg.mode=='forward' or epoch<=cfg.warmup_epochs+cfg.physics_ramp_epochs:
            phase='net'
        else:
            position=(epoch-cfg.warmup_epochs-cfg.physics_ramp_epochs-1)%(cfg.net_steps+cfg.nu_steps)
            phase='net' if position<cfg.net_steps else 'nu'
        if phase=='nu' and cfg.nu_update=='analytic':
            # Macro/neq weights remain frozen; no unnecessary data/wall backward.
            for p in network_params: p.requires_grad_(False)
            model.log_nu.requires_grad_(False)
            rec=analytic_nu_update(model,pool,rng,cfg,device,dtype)
            last=hist[-1] if hist else {}
            row={k:last.get(k,0.0) for k in ('loss','data','mbc','mbc_rho','mbc_uv','fbc','Ebc','physics',
                                               'mass','momentum','kinetic','div','rho_reg','moment','ce')}
            row.update({'epoch':epoch,'phase':'nu_analytic','alpha':alpha,'stage':'full_physics',
                        'nu':float(model.nu.detach()),
                        'time_sec':time.perf_counter()-history_start,**rec})
            hist.append(row)
            if epoch%cfg.print_every==0 or epoch==cfg.epochs:
                print(f'epoch={epoch:06d} phase=nu_analytic nu={row["nu"]:.7g} '
                      f'candidate={rec["nu_candidate"]:.7g} accepted={rec["nu_accepted"]} '
                      f'signal={rec["shear_signal"]:.3e}',flush=True)
            if cfg.save_every>0 and epoch%cfg.save_every==0:
                torch.save({'epoch':epoch,'model':model.state_dict(),
                            'net_opt':optim_net.state_dict(),'nu_opt':optim_nu.state_dict(),
                            'net_sched':scheduler_net.state_dict(),'nu_sched':scheduler_nu.state_dict(),
                            'config':asdict(cfg),'history':hist,'rng_state':rng.bit_generator.state},
                           out/f'checkpoint_{epoch:06d}.pt')
                pd.DataFrame(hist).to_csv(out/'loss_history.csv',index=False)
            continue
        net_update=(phase=='net')
        for p in network_params:p.requires_grad_(net_update)
        model.log_nu.requires_grad_(not net_update and cfg.mode=='inverse')
        optim_net.zero_grad(set_to_none=True);optim_nu.zero_grad(set_to_none=True)
        # Use exactly same observation indices for x and y.
        did=rng.integers(len(obs),size=cfg.batch_data)
        xdat=torch.as_tensor(obs[did],dtype=dtype,device=device)
        ydat=torch.as_tensor(targets[did],dtype=dtype,device=device)
        _,ud,vd=model.macro(xdat)
        # DATA: only measured u,v at randomly sampled observation coordinates.
        # No density penalty and no kinetic boundary target enters data_loss.
        data_loss=((ud-ydat[:,0:1])**2+(vd-ydat[:,1:2])**2).mean()
        walls_loss={'mbc':torch.zeros((),dtype=dtype,device=device),
                    'mbc_rho':torch.zeros((),dtype=dtype,device=device),
                    'mbc_uv':torch.zeros((),dtype=dtype,device=device),
                    'fbc':torch.zeros((),dtype=dtype,device=device),
                    'Ebc':torch.zeros((),dtype=dtype,device=device)}
        if alpha>0:
            for wall_name in WALLS:
                xw=take(walls[wall_name],cfg.batch_wall,rng,device,dtype,needs_grad=True)
                part=wall_loss(model,xw,wall_name,cfg)
                for k in walls_loss:
                    walls_loss[k]=walls_loss[k]+part[k]
        # Else: wall losses remain exactly zero and no boundary network forward
        # or derivatives are performed. Warmup optimizes observed u/v only.
        phy={'physics':torch.zeros((),dtype=dtype,device=device),
             'mass':torch.zeros((),dtype=dtype,device=device),
             'momentum':torch.zeros((),dtype=dtype,device=device),
             'kinetic':torch.zeros((),dtype=dtype,device=device),
             'div':torch.zeros((),dtype=dtype,device=device),
             'rho_reg':torch.zeros((),dtype=dtype,device=device),
             'moment':torch.zeros((),dtype=dtype,device=device)}
        if alpha>0:
            xp=take_physics(pool,cfg.batch_physics,rng,device,dtype,cfg)
            phy=interior_losses(model,xp,cfg)
        ce=torch.zeros((),dtype=dtype,device=device)
        # CE interior bootstrap is optional, and OFF by default (w_ce=0).
        # This does not disable the CE inflow wall loss L_fbc.
        if alpha>0 and cfg.w_ce>0 and alpha<1 and net_update:
            ce=ce_bootstrap_loss(model,xp)
        physics_and_boundary_loss=(
            phy['physics']
            +cfg.w_mbc*walls_loss['mbc']
            +cfg.w_fbc*walls_loss['fbc']
            +cfg.w_Ebc*walls_loss['Ebc']
            +cfg.w_ce*(1-alpha)*ce
        )
        # IMPORTANT: macro BC belongs to the same activated physical group as
        # the PDE and kinetic BCs; phase 1 is exclusively observation data.
        loss=cfg.w_data*data_loss+alpha*physics_and_boundary_loss
        if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite loss at epoch={epoch}')
        loss.backward()
        # One-time fail-fast diagnosis: both KANs must receive finite,
        # nonzero gradients from the kinetic training objective.
        if net_update and alpha>0 and epoch==cfg.warmup_epochs+1:
            for label,params in (('KAN_eq',list(model.macro_net.parameters())),
                                 ('KAN_neq',list(model.neq_net.parameters()))):
                gs=[p.grad for p in params if p.grad is not None]
                if not gs or any(not torch.isfinite(g).all() for g in gs):
                    raise FloatingPointError(f'{label} did not receive finite gradients from physics training')
                peak=max(float(g.detach().abs().max()) for g in gs)
                if peak<=0:
                    raise RuntimeError(f'{label} has zero gradient at physics activation; inspect weights and losses')
                print(f'Gradient check passed for {label}: max_abs_gradient={peak:.3e}',flush=True)
        if net_update:
            nn.utils.clip_grad_norm_(network_params,1.)
            optim_net.step()
            scheduler_net.step()
        elif model.log_nu.grad is not None:
            nn.utils.clip_grad_norm_([model.log_nu],1.)
            optim_nu.step()
            scheduler_nu.step()
            with torch.no_grad():model.log_nu.clamp_(math.log(cfg.nu_min),math.log(cfg.nu_max))
        r={'epoch':epoch,'phase':phase,'loss':float(loss.detach()),'data':float(data_loss.detach()),
           'mbc':float(walls_loss['mbc'].detach()),
           'mbc_rho':float(walls_loss['mbc_rho'].detach()),
           'mbc_uv':float(walls_loss['mbc_uv'].detach()),
           'fbc':float(walls_loss['fbc'].detach()),
           'Ebc':float(walls_loss['Ebc'].detach()),
           **{k:float(v.detach()) for k,v in phy.items()},
           'ce':float(ce.detach()),'nu_candidate':float('nan'),
           'shear_signal':float('nan'),'shear_score_old':float('nan'),
           'shear_score_new':float('nan'),'nu_accepted':0,
           'alpha':alpha,'stage':('data_only' if alpha==0 else ('physics_ramp' if alpha<1 else 'full_physics')),'nu':float(model.nu.detach()),'time_sec':time.perf_counter()-history_start}
        hist.append(r)
        if epoch==1 or epoch%cfg.print_every==0 or epoch==cfg.epochs:
            print(f'epoch={epoch:06d} stage={r["stage"]} phase={phase} total={r["loss"]:.4e} '
                  f'data={r["data"]:.3e} mbc={r["mbc"]:.3e} '
                  f'(rho={r["mbc_rho"]:.2e},uv={r["mbc_uv"]:.2e}) '
                  f'fbc={r["fbc"]:.3e} Ebc={r["Ebc"]:.3e} '
                  f'physics={r["physics"]:.3e} nu={r["nu"]:.7g} ',flush=True)
        if cfg.save_every>0 and epoch%cfg.save_every==0:
            torch.save({'epoch':epoch,'model':model.state_dict(),
                        'net_opt':optim_net.state_dict(),'nu_opt':optim_nu.state_dict(),
                        'net_sched':scheduler_net.state_dict(),'nu_sched':scheduler_nu.state_dict(),
                        'config':asdict(cfg),'history':hist,'rng_state':rng.bit_generator.state},out/f'checkpoint_{epoch:06d}.pt')
            pd.DataFrame(hist).to_csv(out/'loss_history.csv',index=False)
    for p in model.parameters():p.requires_grad_(True)
    pd.DataFrame(hist).to_csv(out/'loss_history.csv',index=False)
    pd.DataFrame([{'epoch':x['epoch'],'nu':x['nu']} for x in hist]).to_csv(out/'nu_history.csv',index=False)
    torch.save({'model':model.state_dict(),'config':asdict(cfg),'epoch':cfg.epochs},
               out/'final_model.pt')
    (out/'training_time.json').write_text(
        json.dumps({'total_seconds':time.perf_counter()-history_start,'epochs':cfg.epochs},
                   indent=2),encoding='utf-8')
    stats=evaluate(model,values,mask,cfg,device,dtype,out)
    profile(model,cfg,pool,device,dtype,out)
    for name,fields in [('loss_history.png',['loss','data','mbc','physics','fbc','Ebc']),
                        ('nu_history.png',['nu'])]:
        fig,ax=plt.subplots(figsize=(7,4.5))
        t=np.array([h['epoch'] for h in hist])
        for key in fields:
            y=np.array([h[key] for h in hist]);
            if key!='nu':y=np.clip(y,1e-14,None)
            ax.plot(t,y,label=key)
        if name=='loss_history.png':ax.set_yscale('log')
        ax.legend();ax.set(xlabel='iteration');fig.tight_layout();fig.savefig(out/name,dpi=160);plt.close(fig)
    stats['nu_update_mode']=cfg.nu_update
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)
    return stats


if __name__=='__main__':
    main(parse())
