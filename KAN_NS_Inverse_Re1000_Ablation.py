"""KAN–Navier–Stokes inverse ablation for the steady lid-driven cavity.

Design lineage:
  * PINN input file ``pinn-反.py``: streamfunction/pressure (psi, p), velocity
    u=psi_y, v=-psi_x, steady incompressible NS residual, learnable viscosity.
  * KAN input file ``MRT-in-nu-Z-all-1000.py``: spline KANLinear with SiLU base
    and cubic B-splines. There is ONE KAN, and NO LBM/MRT/SRT/neq network.
  * Reference: ``num-1000.csv`` (X,Y,u,v), Re=1000, U=L=1, nu_ref=0.001.

Physical residual:
    r_u = u*u_x + v*u_y + p_x - nu*(u_xx+u_yy)
    r_v = u*v_x + v*v_y + p_y - nu*(v_xx+v_yy)
    div = u_x+v_y = 0 by the streamfunction representation.

Important experimental limitation: this baseline changes both the kinetic PDE
and the output representation relative to a (rho,u,v)+(f_neq) two-branch MRT
model. It tests the complete KAN–NS baseline, not an isolated MRT contribution.

Requires: torch, numpy, pandas, matplotlib, scipy.
Examples:
    python KAN_NS_Inverse_Re1000_Ablation.py --data num-1000.csv
    python KAN_NS_Inverse_Re1000_Ablation.py --epochs 200 --width 16 --depth 1 --batch-size 16
    python KAN_NS_Inverse_Re1000_Ablation.py --width 60 --depth 5 --epochs 40000
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # ✅ 防止OpenMP库冲突
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
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import griddata

HERE = Path(__file__).resolve().parent


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class KANLinear(nn.Module):
    """SiLU-base + learnable B-spline edge functions from the supplied KAN code.

    NOTE: hidden tanh activations lie in [-1,1], so all spline grids use
    [-1,1]. The original script passed [0,1] to its KAN layers even though
    hidden activations were tanh; that would leave negative hidden values
    outside the primary B-spline grid. Here the range is consistent.
    """
    def __init__(self, in_features, out_features, grid_size=15,
                 spline_order=3, scale_noise=0.1, scale_base=1.,
                 scale_spline=1., base_activation=nn.SiLU,
                 grid_range=(-1., 1.)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]).expand(in_features, -1).contiguous()
        self.register_buffer('grid', grid)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(
            out_features, in_features, grid_size + spline_order))
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size+1, self.in_features,
                                 self.out_features) - .5) * self.scale_noise
                     / self.grid_size)
            coeff = self.curve2coeff(
                self.grid.T[self.spline_order:-self.spline_order], noise)
            self.spline_weight.copy_(coeff)
            nn.init.kaiming_uniform_(
                self.spline_scaler, a=math.sqrt(5)*self.scale_spline)

    def b_splines(self, x):
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(f'KAN requires [N,{self.in_features}], got {x.shape}')
        grid = self.grid
        xx = x.unsqueeze(-1)
        bases = ((xx >= grid[:, :-1]) & (xx < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order+1):
            bases = ((xx-grid[:, :-(k+1)]) / (grid[:, k:-1]-grid[:, :-(k+1)])
                     * bases[:, :, :-1]
                     + (grid[:, k+1:]-xx) / (grid[:, k+1:]-grid[:, 1:-k])
                     * bases[:, :, 1:])
        return bases.contiguous()

    def curve2coeff(self, x, y):
        a = self.b_splines(x).transpose(0, 1)
        b = y.transpose(0, 1)
        return torch.linalg.lstsq(a, b).solution.permute(2,0,1).contiguous()

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.in_features)
        base = F.linear(self.base_activation(x), self.base_weight)
        scaled_spline = self.spline_weight * self.spline_scaler.unsqueeze(-1)
        spline = F.linear(self.b_splines(x).reshape(x.shape[0], -1),
                          scaled_spline.reshape(self.out_features, -1))
        return (base + spline).reshape(*shape[:-1], self.out_features)


class KAN_PsiP(nn.Module):
    """One KAN: (x,y) -> (psi,p) with physical-input differentiation."""
    def __init__(self, width, depth, grid_size, spline_order):
        super().__init__()
        widths = [2] + [width] * depth + [2]
        self.layers = nn.ModuleList([
            KANLinear(a,b,grid_size=grid_size,spline_order=spline_order,
                      base_activation=nn.SiLU,grid_range=(-1.,1.))
            for a,b in zip(widths[:-1],widths[1:])
        ])

    def forward(self, xy):
        # Data span [0,1]^2: normalized input [-1,1]. Since xy is physical
        # and only this affine transform is used, autograd includes the factor 2.
        z = 2.0 * xy - 1.0
        for j,layer in enumerate(self.layers):
            z = layer(z)
            if j < len(self.layers)-1:
                z = torch.tanh(z)
        return z


class KAN_NS_Inverse(nn.Module):
    def __init__(self, width, depth, grid_size, spline_order, initial_nu):
        super().__init__()
        if initial_nu <= 0:
            raise ValueError('initial_nu must be positive')
        self.net = KAN_PsiP(width,depth,grid_size,spline_order)
        # Match the supplied MLP-PINN: log parameter ensures positive nu.
        self.log_nu = nn.Parameter(torch.tensor(math.log(initial_nu),
                                                dtype=torch.float32))

    @property
    def nu(self):
        return torch.exp(self.log_nu)

    def raw_psi_p(self, xy):
        return self.net(xy)


def grad(y, x, create_graph=True):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                               create_graph=create_graph,
                               retain_graph=True)[0]


def uv_psi_p(model, xy, create_graph=True):
    raw = model.raw_psi_p(xy)
    psi,p = raw[:,0:1], raw[:,1:2]
    gp = grad(psi, xy, create_graph=create_graph)
    return gp[:,1:2], -gp[:,0:1], psi, p


def ns_residuals(model, xy):
    """Steady Navier–Stokes with u=psi_y and v=-psi_x.

    Third derivatives of cubic spline psi appear in the viscous Laplacian;
    spline third derivatives are defined almost everywhere (except knots).
    """
    u,v,_,p = uv_psi_p(model, xy, create_graph=True)
    du = grad(u,xy)
    dv = grad(v,xy)
    dp = grad(p,xy)
    ux,uy = du[:,0:1],du[:,1:2]
    vx,vy = dv[:,0:1],dv[:,1:2]
    lap_u = grad(ux,xy)[:,0:1] + grad(uy,xy)[:,1:2]
    lap_v = grad(vx,xy)[:,0:1] + grad(vy,xy)[:,1:2]
    ru = u*ux + v*uy + dp[:,0:1] - model.nu * lap_u
    rv = u*vx + v*vy + dp[:,1:2] - model.nu * lap_v
    div = ux+vy
    return ru,rv,div


@dataclass
class Config:
    data: str = str(HERE/'num-1000.csv')
    output: str = ''
    epochs: int = 40000  # matches PINN-反.py iteration count
    batch_size: int = 128 # separate 128 per wall, interior and observed data
    n_train: int = 1000
    width: int = 30       # MRT KAN_eq hidden width; use 60 for MLP-like widths
    depth: int = 1        # number of hidden KAN layers
    grid_size: int = 15
    spline_order: int = 3
    initial_nu: float = 0.01  # not equal to Re=1000 truth 0.001
    true_nu: float = 0.001    # POST-TRAINING metric only, never used in losses
    lr: float = 0.001
    nu_lr: float = 0.001
    lr_decay_step: int = 1000
    lr_decay_gamma: float = 0.9
    w_pde: float = 1.0
    w_bc: float = 1.0
    w_data: float = 1.0
    seed: int = 42
    print_every: int = 100
    save_every: int = 5000
    eval_batch_size: int = 256
    prediction_grid_size: int = 100
    noise: float = 0.0  # 0 by default for clean KAN–NS ablation
    deterministic: int = 0
    save_sampling_every: int = 0  # 0: first epoch only; n: also every n epochs


def load_observations(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()
    needed = ['X','Y','u','v']
    if any(k not in data.columns for k in needed):
        raise ValueError(f'Input must contain {needed}; got {list(data.columns)}')
    values = data[needed].to_numpy(np.float32)
    if len(values)==0 or not np.isfinite(values).all():
        raise ValueError('Dataset is empty or contains NaN/Inf values')
    if not np.all((values[:,:2] >= -1e-6) & (values[:,:2] <= 1+1e-6)):
        raise ValueError('This benchmark expects coordinates in [0,1]^2')
    return data, values


def sample_wall(kind, n, device):
    # Avoid lid/corner discontinuities (not imposing contradictory top and
    # stationary-wall constraints on the exact same corner nodes).
    s = 1e-5 + (1.-2e-5) * torch.rand(n,1,device=device)
    z = torch.zeros_like(s)
    o = torch.ones_like(s)
    if kind=='top': xy = torch.cat([s,o],dim=1); uv = torch.cat([o,z],dim=1)
    elif kind=='left': xy = torch.cat([z,s],dim=1); uv = torch.cat([z,z],dim=1)
    elif kind=='right': xy = torch.cat([o,s],dim=1); uv = torch.cat([z,z],dim=1)
    elif kind=='bottom': xy = torch.cat([s,z],dim=1); uv = torch.cat([z,z],dim=1)
    else: raise ValueError(kind)
    return xy.requires_grad_(True),uv


def components(model, train_xy, train_uv, batch, device, config, save_samples=False):
    # True physical NS collocation points, sampled anew each iteration.
    interior = torch.rand(batch,2,device=device,requires_grad=True)
    samples=[]
    if save_samples:
        samples.append(('physics_interior', interior.detach().cpu().numpy()))
    ru,rv,_ = ns_residuals(model,interior)
    loss_ru = ru.square().mean()
    loss_rv = rv.square().mean()
    loss_pde = loss_ru + loss_rv

    # Velocity-only observation loss; no full-field labels used.
    ids = torch.randint(len(train_xy),(batch,),device=device)
    x_data = train_xy[ids].detach().requires_grad_(True)
    u_data = train_uv[ids]
    if save_samples:
        samples.append(('observed_data', x_data.detach().cpu().numpy()))
    u,v,_,_ = uv_psi_p(model,x_data)
    loss_data_u=(u-u_data[:,0:1]).square().mean()
    loss_data_v=(v-u_data[:,1:2]).square().mean()
    # Match PINN-反.py: torch.mean([ (u-u*)^2,(v-v*)^2 ])
    loss_data=(loss_data_u+loss_data_v)*.5

    wall_losses={}
    for name in ('top','left','right','bottom'):
        xb, targets = sample_wall(name,batch,device)
        if save_samples:
            samples.append(('boundary_'+name,xb.detach().cpu().numpy()))
        ub,vb,_,_=uv_psi_p(model,xb)
        pred=torch.cat([ub,vb],dim=1)
        # sum of squared u and v components (same scaling as supplied PINN).
        wall_losses[name] = (pred-targets).square().sum(dim=1).mean()
    loss_bc=sum(wall_losses.values())
    total=(config.w_pde*loss_pde + config.w_bc*loss_bc
           + config.w_data*loss_data)
    result={'total':total,'pde':loss_pde,'pde_u':loss_ru,'pde_v':loss_rv,
            'bc':loss_bc,'data':loss_data,'data_u':loss_data_u,
            'data_v':loss_data_v}
    for name,loss in wall_losses.items(): result['bc_'+name]=loss
    return result, samples


def save_csv(path, rows, columns=None):
    if not rows:
        return
    with open(path,'w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=columns or list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)


def predict_batches(model, xy: np.ndarray, device, batch_size):
    model.eval()
    buf=[]
    for ids in range(0,len(xy),batch_size):
        xb=torch.as_tensor(xy[ids:ids+batch_size],dtype=torch.float32,
                           device=device).detach().requires_grad_(True)
        # Do NOT put this under torch.no_grad(): velocity requires psi derivatives.
        with torch.enable_grad():
            u,v,psi,p=uv_psi_p(model,xb,create_graph=False)
            pred=torch.cat([u,v,psi,p],dim=1).detach().cpu().numpy()
        buf.append(pred)
    return np.concatenate(buf,axis=0)


def error_metrics(pred,ref):
    delta=pred-ref
    return {
        'L2_u':float(np.linalg.norm(delta[:,0])/max(np.linalg.norm(ref[:,0]),1e-12)),
        'L2_v':float(np.linalg.norm(delta[:,1])/max(np.linalg.norm(ref[:,1]),1e-12)),
        'L2_uv':float(np.linalg.norm(delta)/max(np.linalg.norm(ref),1e-12)),
        'MAE_u':float(np.mean(np.abs(delta[:,0]))),
        'MAE_v':float(np.mean(np.abs(delta[:,1]))),
        'RMSE_u':float(np.sqrt(np.mean(delta[:,0]**2))),
        'RMSE_v':float(np.sqrt(np.mean(delta[:,1]**2))),
    }


def save_final(model, data, values, train_ids, noise_uv, history,
               output,device,config,elapsed,peak_memory_mb):
    xy=values[:,:2]
    true=values[:,2:4]
    all_prediction=predict_batches(model,xy,device,config.eval_batch_size)
    inferred=all_prediction[:,:2]
    mask=np.ones(len(values),dtype=bool);mask[train_ids]=False
    hold=error_metrics(inferred[mask],true[mask]) if mask.any() else None
    train=error_metrics(inferred[~mask],true[~mask])
    full=error_metrics(inferred,true)
    nu=float(model.nu.detach().cpu().item())
    relative=100*abs(nu-config.true_nu)/config.true_nu if config.true_nu>0 else None
    metrics={
        'model':'single_KAN_NS_streamfunction_pressure',
        'nu_inferred':nu,
        'nu_reference_post_training':config.true_nu,
        'nu_relative_error_percent':relative,
        're_inferred':1./nu,
        'parameter_count':sum(p.numel() for p in model.parameters()),
        'time_seconds':elapsed,
        'peak_cuda_memory_mb':peak_memory_mb,
        'full_grid':full,'training_subset':train,'heldout_only':hold,
        'train_count':len(train_ids),'heldout_count':int(mask.sum()),
        'noise_fraction':config.noise
    }
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8')
    pd.DataFrame([{'partition':'full_grid',**full},
                  {'partition':'training_subset',**train},
                  *([{'partition':'heldout_only',**hold}] if hold else [])]).to_csv(
                      output/'error_metrics.csv',index=False)
    fields=pd.DataFrame({
        'X':xy[:,0],'Y':xy[:,1],
        'u_reference':true[:,0],'v_reference':true[:,1],
        'u_pred':inferred[:,0],'v_pred':inferred[:,1],
        'u_abs_error':np.abs(true[:,0]-inferred[:,0]),
        'v_abs_error':np.abs(true[:,1]-inferred[:,1]),
        'u_signed_error':inferred[:,0]-true[:,0],
        'v_signed_error':inferred[:,1]-true[:,1],
        'speed_reference':np.linalg.norm(true,axis=1),
        'speed_pred':np.linalg.norm(inferred,axis=1),
        'psi_pred':all_prediction[:,2],
        'p_pred':all_prediction[:,3],
        'is_training_point':(~mask).astype(np.int32)
    })
    fields.to_csv(output/'prediction_vs_reference.csv',index=False)
    fields[['X','Y','u_pred','v_pred']].rename(
        columns={'u_pred':'u','v_pred':'v'}).to_csv(output/'predicted_flow_data.csv',index=False)

    # Regular grid independent of observation file layout.
    n=config.prediction_grid_size
    xs=np.linspace(0,1,n,dtype=np.float32)
    gx,gy=np.meshgrid(xs,xs,indexing='xy')
    gxy=np.column_stack([gx.ravel(),gy.ravel()]).astype(np.float32)
    grid_pred=predict_batches(model,gxy,device,config.eval_batch_size)
    grid_ref=griddata(xy,true,gxy,method='linear')
    pd.DataFrame({'X':gxy[:,0],'Y':gxy[:,1],
                  'u_pred':grid_pred[:,0],'v_pred':grid_pred[:,1],
                  'psi_pred':grid_pred[:,2],'p_pred':grid_pred[:,3],
                  'u_ref':grid_ref[:,0],'v_ref':grid_ref[:,1],
                  'abs_u_error':np.abs(grid_pred[:,0]-grid_ref[:,0]),
                  'abs_v_error':np.abs(grid_pred[:,1]-grid_ref[:,1])}).to_csv(
                    output/'uniform_grid_data.csv',index=False)

    # Correct centerlines: u(x,y=0.5) and v(x=0.5,y).
    line=np.linspace(0,1,201,dtype=np.float32)
    q_u=np.column_stack([line,np.full_like(line,.5)])
    q_v=np.column_stack([np.full_like(line,.5),line])
    vu=predict_batches(model,q_u,device,config.eval_batch_size)
    vv=predict_batches(model,q_v,device,config.eval_batch_size)
    ref_u=griddata(xy,true[:,0],q_u,method='linear')
    ref_v=griddata(xy,true[:,1],q_v,method='linear')
    pd.DataFrame({'coordinate':line,'u_at_y0p5_pred':vu[:,0],
                  'u_at_y0p5_ref':ref_u,
                  'v_at_x0p5_pred':vv[:,1],
                  'v_at_x0p5_ref':ref_v}).to_csv(output/'centerline_profiles.csv',index=False)

    # Compare radial FFT energy on the SAME n*n grid; high k values are
    # illustrative and depend on grid resolution/normalization.
    if np.isfinite(grid_ref).all():
        def spec(f):
            f=np.asarray(f).reshape(n,n).astype(np.float64)
            ft=np.fft.fftshift(np.fft.fft2(f))
            return np.abs(ft)**2/(n*n)**2
        kr=np.fft.fftshift(np.fft.fftfreq(n,d=1./n))
        kx,ky=np.meshgrid(kr,kr)
        rad=np.floor(np.sqrt(kx*kx+ky*ky)).astype(int)
        rows=[]
        en_ref=spec(grid_ref[:,0])+spec(grid_ref[:,1])
        en_pre=spec(grid_pred[:,0])+spec(grid_pred[:,1])
        for k in range(int(rad.max())+1):
            which=(rad==k)
            if which.any():
                rows.append({'k':k,'energy_reference':float(en_ref[which].sum()),
                             'energy_KAN_NS':float(en_pre[which].sum()),
                             'number_of_modes':int(which.sum())})
        spectra=pd.DataFrame(rows)
        spectra.to_csv(output/'radial_fft_energy.csv',index=False)
        fig,ax=plt.subplots(figsize=(7,4.5))
        ss=spectra.loc[spectra['k']>0]
        ax.semilogy(ss['k'],ss['energy_reference'].clip(lower=1e-18),label='Reference')
        ax.semilogy(ss['k'],ss['energy_KAN_NS'].clip(lower=1e-18),label='KAN–NS')
        ax.set(xlabel='Radial wavenumber k',ylabel='Summed FFT energy',
               title='Velocity spectrum (equal-resolution grid)')
        ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(output/'radial_fft_energy.png',dpi=300);plt.close(fig)

    torch.save({'model_state':model.state_dict(),
                'config':asdict(config), 'metrics':metrics},output/'final_model.pt')
    pd.DataFrame(history).to_csv(output/'loss_history.csv',index=False)
    pd.DataFrame([{'epoch':h['epoch'],'nu':h['nu'],
                   'nu_relative_error_percent':h['nu_relative_error_percent']}
                  for h in history]).to_csv(output/'nu_history.csv',index=False)

    def plot_map(key,title,filename,cmap):
        fig,ax=plt.subplots(figsize=(6.2,5.3))
        # griddata already resampled on a regular mesh
        ar=fields[key].to_numpy()
        f=ax.tricontourf(xy[:,0],xy[:,1],ar,levels=50,cmap=cmap)
        fig.colorbar(f,ax=ax)
        ax.set(xlabel='x',ylabel='y',title=title,aspect='equal')
        fig.tight_layout(); fig.savefig(output/filename,dpi=300);plt.close(fig)
    for key,title,name,cmap in (
        ('u_pred','KAN–NS predicted u','u_pred.png','jet'),
        ('v_pred','KAN–NS predicted v','v_pred.png','jet'),
        ('u_abs_error','|u prediction error|','u_abs_error.png','viridis'),
        ('v_abs_error','|v prediction error|','v_abs_error.png','viridis')):
        plot_map(key,title,name,cmap)

    hist=pd.DataFrame(history)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for col in ('total','pde','bc','data'):
        ax.semilogy(hist['epoch'],hist[col].clip(lower=1e-16),label=col)
    ax.set(xlabel='Epoch',ylabel='Loss',title='KAN–NS loss components')
    ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(output/'training_loss.png',dpi=300);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.plot(hist['epoch'],hist['nu'],label='Inferred nu')
    if config.true_nu>0: ax.axhline(config.true_nu,ls='--',c='red',label='True nu (evaluation only)')
    ax.set(xlabel='Epoch',ylabel='Viscosity',title='KAN–NS viscosity inversion')
    ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(output/'viscosity_history.png',dpi=300);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.2))
    axs[0].plot(line,ref_u,'k-',lw=1.8,label='Reference')
    axs[0].plot(line,vu[:,0],ls='--',label='KAN–NS')
    axs[0].set(xlabel='x',ylabel='u(x,0.5)',title='Horizontal midline')
    axs[1].plot(line,ref_v,'k-',lw=1.8,label='Reference')
    axs[1].plot(line,vv[:,1],ls='--',label='KAN–NS')
    axs[1].set(xlabel='y',ylabel='v(0.5,y)',title='Vertical midline')
    for ax in axs:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'centerline_profiles.png',dpi=300);plt.close(fig)
    return metrics


def run(config:Config):
    set_seed(config.seed)
    if config.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    if not 0<=config.noise<=1:raise ValueError('noise must be in [0,1]')
    if config.epochs<1 or config.batch_size<1 or config.n_train<1:
        raise ValueError('epochs, batch_size, n_train must be >=1')
    if config.depth<1 or config.width<1 or config.grid_size<4:
        raise ValueError('depth and width must be >=1; grid_size >=4')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset,values=load_observations(config.data)
    n=min(len(values),config.n_train)
    rng=np.random.default_rng(config.seed)
    train_ids=np.sort(rng.choice(len(values),size=n,replace=False))
    clean=values[train_ids,2:4].copy()
    noises=np.column_stack([
        config.noise*np.std(clean[:,0])*rng.standard_normal(n),
        config.noise*np.std(clean[:,1])*rng.standard_normal(n)
    ]).astype(np.float32)
    train_uv=clean+noises
    train_xy=values[train_ids,:2].copy()
    out=Path(config.output) if config.output else (HERE/'results_KAN_NS'/
        ('Re1000_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(asdict(config),indent=2,ensure_ascii=False),encoding='utf-8')
    pd.DataFrame({'source_row_index':train_ids,'X':train_xy[:,0],'Y':train_xy[:,1],
                  'u_clean':clean[:,0],'v_clean':clean[:,1],
                  'u_train':train_uv[:,0],'v_train':train_uv[:,1],
                  'added_noise_u':noises[:,0],'added_noise_v':noises[:,1]}).to_csv(
                      out/'training_observations.csv',index=False)
    rest=np.ones(len(values),dtype=bool);rest[train_ids]=False
    pd.DataFrame(values[rest],columns=['X','Y','u','v']).to_csv(
        out/'heldout_observations.csv',index=False)
    (out/'dataset_info.json').write_text(json.dumps({
        'rows':len(values),'selected_training_points':n,
        'heldout_points':int(rest.sum()),'x_range':[float(values[:,0].min()),float(values[:,0].max())],
        'y_range':[float(values[:,1].min()),float(values[:,1].max())],
        'u_std_training':float(np.std(clean[:,0])),
        'v_std_training':float(np.std(clean[:,1])),
        'noise_fraction':config.noise,'note':'True nu is post-training only; no reference nu or p used in losses.'
    },indent=2,ensure_ascii=False),encoding='utf-8')
    x_train=torch.as_tensor(train_xy,device=device)
    y_train=torch.as_tensor(train_uv,device=device)
    model=KAN_NS_Inverse(config.width,config.depth,config.grid_size,
                          config.spline_order,config.initial_nu).to(device)
    param_count=sum(q.numel() for q in model.parameters())
    optimizer=torch.optim.Adam([
        {'params':list(model.net.parameters()), 'lr':config.lr},
        {'params':[model.log_nu], 'lr':config.nu_lr}
    ])
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=config.lr_decay_gamma)
    print(f'Device={device}; Re=1000; train={n}; heldout={int(rest.sum())}; '\
          f'KAN widths={[2]+[config.width]*config.depth+[2]}; parameters={param_count}; '\
          f'initial nu={config.initial_nu}; ref nu used only in evaluation.',flush=True)
    print(f'Output: {out}',flush=True)
    hist=[]
    t0=time.perf_counter()
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1,config.epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        record=(epoch==1 or (config.save_sampling_every>0
                             and epoch%config.save_sampling_every==0))
        parts,sample_records=components(model,x_train,y_train,config.batch_size,
                                       device,config,save_samples=record)
        if record:
            pd.DataFrame([{'epoch':epoch,'source':label,'X':float(x), 'Y':float(y)}
                          for label,pts in sample_records for x,y in pts]).to_csv(
                              out/f'sampling_points_epoch_{epoch:06d}.csv',index=False)
        if not torch.isfinite(parts['total']):
            pd.DataFrame(hist).to_csv(out/'loss_history.csv',index=False)
            raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
        parts['total'].backward()
        nu_gradient=(float(model.log_nu.grad.detach().cpu())
                     if model.log_nu.grad is not None else float('nan'))
        optimizer.step()
        if config.lr_decay_step>0 and epoch%config.lr_decay_step==0:
            scheduler.step()
        nu=float(model.nu.detach().cpu())
        h={'epoch':epoch,**{k:float(v.detach().cpu()) for k,v in parts.items()},
           'nu':nu,'nu_relative_error_percent':
                100.*abs(nu-config.true_nu)/config.true_nu if config.true_nu>0 else float('nan'),
           'nu_log_grad':nu_gradient,'lr':optimizer.param_groups[0]['lr'],
           'elapsed_seconds':time.perf_counter()-t0}
        hist.append(h)
        if epoch==1 or epoch%config.print_every==0 or epoch==config.epochs:
            print(f'Epoch {epoch:06d}/{config.epochs} total={h["total"]:.3e} '
                  f'PDE={h["pde"]:.3e} BC={h["bc"]:.3e} '
                  f'data={h["data"]:.3e} nu={nu:.8f} '
                  f't={h["elapsed_seconds"]:.1f}s',flush=True)
        if config.save_every>0 and epoch%config.save_every==0:
            torch.save({'epoch':epoch,'model_state':model.state_dict(),
                        'optimizer_state':optimizer.state_dict(),
                        'scheduler_state':scheduler.state_dict(),
                        'config':asdict(config)},out/f'checkpoint_{epoch:06d}.pt')
            pd.DataFrame(hist).to_csv(out/'loss_history.csv',index=False)
    elapsed=time.perf_counter()-t0
    peak_mb=(torch.cuda.max_memory_allocated(device)/1024**2) if device.type=='cuda' else 0.
    metrics=save_final(model,dataset,values,train_ids,noises,hist,out,device,
                       config,elapsed,peak_mb)
    print('Training finished. Metrics:',json.dumps(metrics,indent=2,ensure_ascii=False),flush=True)
    return metrics


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for key,field in Config.__dataclass_fields__.items():
        default=field.default
        parser.add_argument('--'+key.replace('_','-'),default=default,type=type(default))
    return Config(**vars(parser.parse_args()))


if __name__=='__main__':
    run(parse_args())
