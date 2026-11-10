#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MLP–MRT / Fourier-feature MLP–MRT / adaptive-weight MLP–MRT inverse ablations.

Source-matched to the uploaded PINN-MRT code `MRT-in-nu-Z-all-1000(2).py`:
  * Two independently parameterized MLP branches, Eq=(rho,u,v), Neq=(9 fneq)
  * The exact D2Q9 equilibrium, MRT relaxation rates and boundary direction sets
  * Default nn.Linear PyTorch initialization (optionally Xavier)
  * Existing L_pde + L_data + L_bc structure, Adam / LBFGS optimizer
  * Optional Fourier feature input, homoscedastic adaptive loss weights.

The same model also offers `--physics-mode consistent` as a separate explicitly
labelled experiment, but never mix source and consistent results in an ablation.

ONE invocation trains ONE model, set --baseline plain|fourier|adaptive|fourier_adaptive.

IMPORTANT, physics version:
  --physics-mode source (DEFAULT) explicitly reproduces the available KAN-MRT
  implementation's row-wise collision multiplication and boundary formula,
  including hardcoded boundary tau=0.503 (a source-code limitation).
  --physics-mode consistent uses the conventional M matrix moment ordering,
  differentiable tau(nu), physical row-wise collision C.T, and first-order
  CE approximation fneq* = - C^{-1} (xi·grad feq) at selected boundaries.
Do not compare models trained using different physics modes as isolated ablations.

Data observed at 1000 points; Re=1000 reference nu=0.001 used ONLY for metrics.
No claims of accuracy are made until the full training is performed.
"""
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # ✅ 防止OpenMP库冲突
import argparse
import csv
import json
import math
import random
import time
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

ROOT = Path(__file__).resolve().parent
XI_DEF = [[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]]
W_DEF = [4/9]+[1/9]*4+[1/36]*4
M_DEF = [
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
MINV_DEF = [
 [1/9,-1/9,1/9,0,0,0,0,0,0],
 [1/9,-1/36,-1/18,1/6,-1/6,0,0,1/4,0],
 [1/9,-1/36,-1/18,0,0,1/6,-1/6,-1/4,0],
 [1/9,-1/36,-1/18,-1/6,1/6,0,0,1/4,0],
 [1/9,-1/36,-1/18,0,0,-1/6,1/6,-1/4,0],
 [1/9,1/18,1/36,1/6,1/12,1/6,1/12,0,1/4],
 [1/9,1/18,1/36,-1/6,-1/12,1/6,1/12,0,-1/4],
 [1/9,1/18,1/36,-1/6,-1/12,-1/6,-1/12,0,1/4],
 [1/9,1/18,1/36,1/6,1/12,-1/6,-1/12,0,-1/4],
]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def network_parameter_count(input_size, hidden, depth, output_size):
    dimensions=[input_size]+[hidden]*depth+[output_size]
    return sum((a+1)*b for a,b in zip(dimensions[:-1],dimensions[1:]))


def chosen_width(args):
    if not args.match_params or 'fourier' not in args.baseline:
        return args.hidden
    # Match total trainable neural-network parameter count, excluding viscosity
    # and 3 additional adaptive loss-weight parameters.
    base_target = (network_parameter_count(2,args.hidden,args.num_hidden,3) +
                   network_parameter_count(2,args.hidden,args.num_hidden,9))
    in_f=2+2*args.fourier_dim if args.include_raw else 2*args.fourier_dim
    return min(range(2,512),key=lambda h: abs(
        network_parameter_count(in_f,h,args.num_hidden,3)+
        network_parameter_count(in_f,h,args.num_hidden,9)-base_target))


class FourierFeatures(nn.Module):
    """Fixed random Fourier features: [x,y,sin(2π Bx),cos(2π Bx)]."""
    def __init__(self, dim=16, scale=1.0, seed=314159, include_raw=True):
        super().__init__()
        rng=np.random.default_rng(seed)
        B=rng.normal(loc=0.0,scale=scale,size=(dim,2)).astype(np.float32)
        self.register_buffer('B',torch.tensor(B))
        self.include_raw=include_raw

    @property
    def output_dim(self):
        return self.B.shape[0]*2+(2 if self.include_raw else 0)

    def forward(self,xy):
        arg=2.0*math.pi*(xy @ self.B.T)
        vals=[torch.sin(arg),torch.cos(arg)]
        if self.include_raw:
            vals.insert(0,xy)
        return torch.cat(vals,dim=1)


class MLP(nn.Module):
    def __init__(self,input_dim,hidden,depth,output_dim,init_method='source'):
        super().__init__()
        layers=[]
        a=input_dim
        for _ in range(depth):
            fc=nn.Linear(a,hidden)
            if init_method=='xavier':
                nn.init.xavier_normal_(fc.weight)
                nn.init.zeros_(fc.bias)
            # source: do not override nn.Linear's PyTorch default initializer
            layers.extend((fc,nn.Tanh()))
            a=hidden
        fc=nn.Linear(a,output_dim)
        if init_method=='xavier':
            nn.init.xavier_normal_(fc.weight)
            nn.init.zeros_(fc.bias)
        layers.append(fc)
        self.net=nn.Sequential(*layers)

    def forward(self,x):
        return self.net(x)


class DualMLPMRT(nn.Module):
    def __init__(self, args):
        super().__init__()
        fourier=('fourier' in args.baseline)
        self.features=(FourierFeatures(args.fourier_dim,args.fourier_scale,
                          args.fourier_seed,args.include_raw) if fourier else nn.Identity())
        self.hidden=chosen_width(args)
        in_dim=self.features.output_dim if fourier else 2
        self.eq=MLP(in_dim,self.hidden,args.num_hidden,3,args.init_method)
        self.neq=MLP(in_dim,self.hidden,args.num_hidden,9,args.init_method)
        self.nu_hat=nn.Parameter(torch.tensor(float(args.nu_init),dtype=torch.float32))
        self.logvars=nn.Parameter(torch.zeros(3),requires_grad=('adaptive' in args.baseline))
        self.is_adaptive=('adaptive' in args.baseline)

    def outputs(self,xy):
        f=self.features(xy)
        return self.eq(f),self.neq(f)

    def macroscopic(self,xy):
        return self.eq(self.features(xy))

    def tau(self):
        return 3.0*self.nu_hat+0.5

    def weighted_loss(self,pde,data,bc):
        component=torch.stack((pde,data,bc))
        if self.is_adaptive:
            # Homoscedastic trainable log-weight formulation. The original
            # PDE residual uses RMSE, so these are regularized adaptive
            # task weights, not calibrated Gaussian noise variances.
            return torch.sum(torch.exp(-self.logvars)*component+self.logvars)
        return component.sum()

    def weights(self):
        return torch.exp(-self.logvars.detach()).cpu().numpy() if self.is_adaptive else np.ones(3)


class MRTPhysics(nn.Module):
    def __init__(self, mode='source'):
        super().__init__()
        self.mode=mode
        self.register_buffer('xi',torch.tensor(XI_DEF,dtype=torch.float32))
        self.register_buffer('w',torch.tensor(W_DEF,dtype=torch.float32))
        self.register_buffer('M',torch.tensor(M_DEF,dtype=torch.float32))
        self.register_buffer('Minv',torch.tensor(MINV_DEF,dtype=torch.float32))
        if not torch.allclose(self.Minv@self.M,torch.eye(9),atol=1e-5):
            raise RuntimeError('Provided MRT moment matrix and inverse are inconsistent')

    def equilibrium(self,rho,u,v):
        U=torch.cat([u,v],dim=1)
        eu=U@self.xi.T
        return self.w[None,:]*rho*(1.+3.*eu+4.5*eu.square()-1.5*U.square().sum(dim=1,keepdim=True))

    def collision_matrix(self, tau, boundary=False):
        # source PDE rates differ from source boundary rates; retained when
        # physics-mode=source to isolate architecture rather than fix physics.
        if self.mode=='source' and not boundary:
            rates=[1.,0.8,0.8,1.,1.,1.1,1.1]
        else:
            rates=[1.,0.8,0.8,1.,1.1,1.,1.1]
        vals=torch.cat((torch.tensor(rates,device=tau.device,dtype=tau.dtype),
                        (1.0/tau).reshape(1).expand(2)))
        return self.Minv@torch.diag(vals)@self.M

    @staticmethod
    def coord_grad(field,X):
        return torch.autograd.grad(field,X,torch.ones_like(field),create_graph=True)[0]

    def advection(self,f,X):
        components=[]
        for i in range(9):
            deriv=self.coord_grad(f[:,i:i+1],X)
            components.append((deriv*self.xi[i]).sum(dim=1,keepdim=True))
        return torch.cat(components,dim=1)

    def collision_rows(self,fneq,C):
        return fneq @ (C if self.mode=='source' else C.T)

    def pde_loss(self, X, model):
        X=X.detach().requires_grad_(True)
        macro, fneq=model.outputs(X)
        feq=self.equilibrium(macro[:,0:1],macro[:,1:2],macro[:,2:3])
        f=feq+fneq
        C=self.collision_matrix(model.tau(),boundary=False)
        res=self.advection(f,X)+self.collision_rows(fneq,C)
        return (res.square().mean()+1e-10).sqrt()

    def data_loss(self,model,obs):
        pred=model.macroscopic(obs[:,:2])[:,1:3]
        return ((pred-obs[:,2:4]).square().sum(dim=1)).mean()

    def boundary_loss(self,model,batches):
        total=0.0
        for xy, uv_target, normal in batches:
            X=xy.detach().requires_grad_(True)
            macro,fneq=model.outputs(X)
            rho=macro[:,0:1]
            u=macro[:,1:2]
            v=macro[:,2:3]
            feq=self.equilibrium(rho,u,v)
            f=feq+fneq
            l_mbc=((rho-1.).square().mean()+
                  (u-uv_target[:,0:1]).square().mean()+
                  (v-uv_target[:,1:2]).square().mean())
            if self.mode=='source':
                # Source's hardcoded tau=0.503 and its original boundary
                # formula are retained only for exact source comparability.
                C_b=self.collision_matrix(torch.tensor(0.503,device=X.device),boundary=True)
            else:
                C_b=self.collision_matrix(model.tau(),boundary=True)
            dot=self.xi@normal
            incoming=torch.where(dot>0)[0].tolist()
            outgoing=torch.where(dot<=0)[0].tolist()
            # Need all directions for MRT CE inverse due mode coupling.
            adv_eq=self.advection(feq,X)
            if self.mode=='source':
                # literal source algorithm: loop accumulates incoming advection
                adv_partial=torch.zeros_like(adv_eq)
                l_fbc=0.0
                for i in incoming:
                    # avoid autograd-breaking inplace replacement
                    keep=torch.zeros((9,),device=X.device,dtype=X.dtype)
                    keep[i]=1.0
                    adv_partial=adv_partial+adv_eq[:,i:i+1]*keep[None,:]
                    fneq_star=-(adv_partial@C_b)
                    l_fbc=l_fbc+(fneq[:,i]-fneq_star[:,i]).square().mean()
            else:
                # Conventional steady first-order MRT CE approximation
                # fneq(col) = -C_b^{-1} a(col)  =>  fneq(row)=-a(row) C_b^{-T}
                fneq_star=-(adv_eq@torch.linalg.inv(C_b).T)
                l_fbc=sum((fneq[:,i]-fneq_star[:,i]).square().mean() for i in incoming)
            # Source outgoing condition is R_j = xi grad f_j + (fneq@C)_j;
            # keep the same selected discrete-velocity normal convention.
            adv_f=self.advection(f,X)
            coll=self.collision_rows(fneq,C_b)
            l_ebc=sum((adv_f[:,i]+coll[:,i]).square().mean() for i in outgoing)
            total=total+l_mbc+l_fbc+l_ebc
        return total


def load_reference(path):
    path=Path(path)
    if not path.exists():
        alternative=ROOT/path
        if alternative.exists(): path=alternative
        else: raise FileNotFoundError(f'Cannot find reference CSV: {path}')
    df=pd.read_csv(path)
    df.columns=df.columns.str.strip()
    required={'X','Y','u','v'}
    if not required.issubset(df.columns):
        raise ValueError(f'CSV must have columns {required}; got {df.columns.tolist()}')
    df=df.dropna(subset=['X','Y','u','v'])
    return df, path.resolve()


def make_sample_pools(df,args,device):
    rng=np.random.default_rng(args.sample_seed)
    xy=df[['X','Y']].to_numpy(dtype=np.float32)
    uv=df[['u','v']].to_numpy(dtype=np.float32)
    if args.n_obs>len(df):
        raise ValueError(f'--n-obs {args.n_obs} > available data {len(df)}')
    idx=rng.choice(len(df),size=args.n_obs,replace=False)
    obs=np.concatenate([xy[idx],uv[idx]],axis=1)
    xy_f=rng.uniform(0,1,size=(args.n_f,2)).astype(np.float32)
    walls=[]
    # Sample each side with the same generator and same order as source.
    wall_defs=[('top',[0.,1.],[1.,0.],[0.,1.]),
               ('left',[0.,0.],[0.,0.],[-1.,0.]),
               ('right',[1.,0.],[0.,0.],[1.,0.]),
               ('bottom',[0.,0.],[0.,0.],[0.,-1.])]
    for name,corner,velocity,n in wall_defs:
        coord=rng.uniform(0,1,size=args.n_b).astype(np.float32)
        points=np.zeros((args.n_b,2),dtype=np.float32)
        if name in ('top','bottom'):
            points[:,0]=coord
            points[:,1]=corner[1]
        else:
            points[:,0]=corner[0]
            points[:,1]=coord
        target=np.repeat(np.array(velocity,dtype=np.float32)[None,:],args.n_b,axis=0)
        walls.append((name,torch.from_numpy(points).to(device),
                      torch.from_numpy(target).to(device),
                      torch.tensor(n,device=device,dtype=torch.float32)))
    return (torch.from_numpy(obs).to(device),torch.from_numpy(xy_f).to(device),walls,idx,obs,xy_f)


def select_batch(pool,size,rg):
    if size<=0 or size>=pool.shape[0]: return pool
    ids=rg.choice(pool.shape[0],size=size,replace=False)
    return pool[torch.as_tensor(ids,device=pool.device,dtype=torch.long)]


def get_training_batches(obs,phys,walls,args,rg):
    obs_b=select_batch(obs,args.batch_data,rg)
    phys_b=select_batch(phys,args.batch_f,rg)
    wall_b=[(select_batch_pair(x,u,args.batch_b,rg),n) for _,x,u,n in walls]
    return obs_b,phys_b,[(a,b,n) for (a,b),n in wall_b]


def select_batch_pair(x,y,size,rg):
    if size<=0 or size>=len(x): return x,y
    ids=torch.as_tensor(rg.choice(len(x),size=size,replace=False),device=x.device,dtype=torch.long)
    return x[ids],y[ids]


def tensor_metric(a,b):
    a=np.asarray(a,dtype=np.float64).reshape(-1)
    b=np.asarray(b,dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-12))


def export_results(out,model,df,obs,phys,walls,args,history,train_seconds,adam_seconds,lbfgs_seconds,device):
    out.mkdir(parents=True,exist_ok=True)
    xy=df[['X','Y']].to_numpy(dtype=np.float32)
    model.eval()
    pred=[]
    with torch.no_grad():
        for begin in range(0,len(xy),args.eval_batch):
            pred.append(model.macroscopic(torch.from_numpy(xy[begin:begin+args.eval_batch]).to(device)).cpu().numpy())
    m=np.concatenate(pred,axis=0)
    result=df[['X','Y','u','v']].copy()
    result['rho_pred']=m[:,0]
    result['u_pred']=m[:,1]
    result['v_pred']=m[:,2]
    result['abs_error_u']=np.abs(result['u']-result['u_pred'])
    result['abs_error_v']=np.abs(result['v']-result['v_pred'])
    result['speed_pred']=np.hypot(result['u_pred'],result['v_pred'])
    result['p_pred']=result['rho_pred']/3.0
    result.to_csv(out/'prediction_vs_reference.csv',index=False)
    pd.DataFrame({'X':result.X,'Y':result.Y,'u':result.u_pred,'v':result.v_pred,
                  'rho':result.rho_pred,'speed':result.speed_pred,'p':result.p_pred}).to_csv(out/'grid_data.csv',index=False)
    pd.DataFrame(obs.detach().cpu().numpy(),columns=['X','Y','u','v']).to_csv(out/'training_observations.csv',index=False)
    pd.DataFrame(phys.detach().cpu().numpy(),columns=['X','Y']).to_csv(out/'physics_points.csv',index=False)
    for name,x,t,n in walls:
        pd.DataFrame(np.c_[x.cpu().numpy(),t.cpu().numpy()],columns=['X','Y','u_bc','v_bc']).to_csv(out/f'boundary_{name}.csv',index=False)
    pred_nu=float(model.nu_hat.detach().cpu())
    metrics={'baseline':args.baseline,'physics_mode':args.physics_mode,'seed':args.seed,
             'init_method':args.init_method,
             'sample_seed':args.sample_seed,'nu_init':args.nu_init,
             'nu_pred':pred_nu,'nu_true_eval_only':args.true_nu,
             'nu_relative_error_percent':abs(pred_nu-args.true_nu)/args.true_nu*100 if args.true_nu>0 else None,
             'L2_u':tensor_metric(result.u,result.u_pred),
             'L2_v':tensor_metric(result.v,result.v_pred),
             'train_seconds':train_seconds,'adam_seconds':adam_seconds,'lbfgs_seconds':lbfgs_seconds,
             'params_total':sum(p.numel() for p in model.parameters() if p.requires_grad),
             'params_eq':sum(p.numel() for p in model.eq.parameters()),
             'params_neq':sum(p.numel() for p in model.neq.parameters()),
             'hidden_effective':model.hidden,'adaptive_weights_pde_data_bc':model.weights().tolist()}
    (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf8')
    pd.DataFrame([metrics]).to_csv(out/'error_metrics.csv',index=False)
    h=pd.DataFrame(history)
    h.to_csv(out/'loss_history.csv',index=False)
    pd.DataFrame({'iteration':h['step'],'nu_hat':h['nu_hat']}).to_csv(out/'nu_history.csv',index=False)
    # Center lines via model evaluation, not by selecting x=0.5 from even-size grid.
    tt=np.linspace(0,1,201,dtype=np.float32)
    xcenter=np.c_[np.full_like(tt,.5),tt]
    ycenter=np.c_[tt,np.full_like(tt,.5)]
    with torch.no_grad():
        cu=model.macroscopic(torch.from_numpy(xcenter).to(device)).cpu().numpy()
        cv=model.macroscopic(torch.from_numpy(ycenter).to(device)).cpu().numpy()
    pd.DataFrame({'y_coord':tt,'u_at_x=0.5':cu[:,1],
                  'x_coord':tt,'v_at_y=0.5':cv[:,2]}).to_csv(out/'centerline_data.csv',index=False)
    if len(h):
        fig,axs=plt.subplots(1,2,figsize=(11,4))
        for k in ('loss_pde','loss_data','loss_bc','raw_sum'):
            axs[0].semilogy(h.step,np.maximum(h[k].to_numpy(),1e-15),label=k)
        axs[0].set_xlabel('Adam/LBFGS logging step')
        axs[0].set_ylabel('Raw loss components')
        axs[0].legend(fontsize=8)
        axs[1].plot(h.step,h.nu_hat,label='inferred viscosity')
        if args.true_nu>0: axs[1].axhline(args.true_nu,ls='--',color='k',label='reference nu')
        axs[1].set_xlabel('Adam/LBFGS logging step')
        axs[1].legend()
        fig.tight_layout(); fig.savefig(out/'loss_and_viscosity.png',dpi=250);plt.close(fig)
        if model.is_adaptive:
            fig,ax=plt.subplots(figsize=(6,4))
            for k in ('w_pde','w_data','w_bc'):ax.plot(h.step,h[k],label=k)
            ax.set_yscale('log');ax.legend();fig.tight_layout()
            fig.savefig(out/'adaptive_weights.png',dpi=250);plt.close(fig)
    # Grid plots if the reference CSV is a rectangular tensor grid.
    xs=np.unique(result.X.to_numpy())
    ys=np.unique(result.Y.to_numpy())
    if len(xs)*len(ys)==len(result):
        ordered=result.sort_values(['Y','X'])
        fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
        for r,(real,pred,label) in enumerate([('u','u_pred','u'),('v','v_pred','v')]):
            gt=ordered[real].to_numpy().reshape(len(ys),len(xs))
            pd_=ordered[pred].to_numpy().reshape(len(ys),len(xs))
            err=np.abs(gt-pd_)
            for c,arr,sub in [(0,gt,'Reference'),(1,pd_,'Prediction'),(2,err,'Absolute error')]:
                im=axes[r,c].imshow(arr,origin='lower',extent=(xs.min(),xs.max(),ys.min(),ys.max()),aspect='equal')
                axes[r,c].set_title(label+' '+sub)
                fig.colorbar(im,ax=axes[r,c],shrink=.8)
        fig.savefig(out/'velocity_error_maps.png',dpi=250);plt.close(fig)
    torch.save({'state_dict':model.state_dict(),'config':vars(args),'metrics':metrics},out/'final_model.pt')
    print('FINAL METRICS',json.dumps(metrics,ensure_ascii=False,indent=2))
    return metrics


def run(args):
    if args.epochs<=0: raise ValueError('--epochs must be positive')
    if args.num_hidden<=0 or args.hidden<=0: raise ValueError('hidden layers and width must be positive')
    if args.physics_mode not in ('source','consistent'):raise ValueError('invalid physics mode')
    if args.init_method not in ('source','xavier'):raise ValueError('invalid init method')
    device=torch.device('cuda' if torch.cuda.is_available() and args.device!='cpu' else 'cpu')
    if args.device=='cuda' and device.type!='cuda':
        raise RuntimeError('CUDA was requested but is unavailable')
    # Reproducible data and Fourier feature sample (independent from network init)
    seed_everything(args.seed)
    df,data_path=load_reference(args.data)
    obs,phys,walls,indices,obs_np,xy_f=make_sample_pools(df,args,device)
    model=DualMLPMRT(args).to(device)
    physics=MRTPhysics(args.physics_mode).to(device)
    wsum={k:float(p.numel()) for k,p in model.named_parameters() if p.requires_grad}
    print(f'DEVICE={device} DATA={data_path} BASELINE={args.baseline} PHYSICS={args.physics_mode}')
    print(f'MLP hidden width={model.hidden}; training parameters={sum(int(v) for v in wsum.values())}')
    print(f'INIT_METHOD={args.init_method} nu_init={args.nu_init:g} seed={args.seed} sample_seed={args.sample_seed}')
    if args.physics_mode=='source':
        print('WARNING: source copies the uploaded PINN-MRT and previous KAN-MRT implementation conventions')
        print('  including boundary tau=0.503, boundary/PDE MRT rate-list mismatch, and row-wise C multiplication.')
        print('  This is a SOURCE-MATCHED ablation, not a verified corrected MRT discretization.')
    if args.batch_f != 0 or args.batch_b != 0 or args.batch_data not in (0,args.n_obs):
        print('NOTE: minibatch training differs from source full-batch optimization; match batches across ALL models.')
    output=(Path(args.output) if args.output else ROOT/'MLP_MRT_ablation_results'/args.physics_mode/args.baseline/f'seed_{args.seed}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    output.mkdir(parents=True,exist_ok=True)
    # Exact original-data row indices, so the training-observation split can be audited.
    pd.DataFrame({'source_row_index_zero_based':indices}).to_csv(output/'observation_indices.csv',index=False)
    # Write setup even for interrupted runs.
    (output/'config.json').write_text(json.dumps(dict(vars(args),data_resolved=str(data_path),device=str(device),
         effective_hidden=model.hidden),ensure_ascii=False,indent=2),encoding='utf8')
    optimizer=torch.optim.Adam([{'params':list(model.eq.parameters())+list(model.neq.parameters()),'lr':args.lr},
                                {'params':[model.nu_hat],'lr':args.nu_lr}]+
                               ([{'params':[model.logvars],'lr':args.weight_lr}] if model.is_adaptive else []))
    history=[]
    rg=np.random.default_rng(args.batch_seed)
    t0=time.perf_counter()
    for it in range(args.epochs):
        model.train()
        ob,ph,wall=get_training_batches(obs,phys,walls,args,rg)
        optimizer.zero_grad(set_to_none=True)
        lp=physics.pde_loss(ph,model)
        ld=physics.data_loss(model,ob)
        lb=physics.boundary_loss(model,wall)
        objective=model.weighted_loss(lp,ld,lb)
        if not torch.isfinite(objective):
            raise FloatingPointError(f'Nonfinite loss at epoch {it}; consider lowering Fourier scale or LR')
        objective.backward()
        # Source uses separate gradient norm clipping for Eq and Neq branches.
        # nu_hat belongs to Eq and logvars are clipped independently.
        torch.nn.utils.clip_grad_norm_(list(model.eq.parameters())+[model.nu_hat],args.clip_grad)
        torch.nn.utils.clip_grad_norm_(model.neq.parameters(),args.clip_grad)
        if model.is_adaptive:
            torch.nn.utils.clip_grad_norm_([model.logvars],args.clip_grad)
        optimizer.step()
        if model.is_adaptive:
            with torch.no_grad(): model.logvars.clamp_(-args.logvar_limit,args.logvar_limit)
        if it%args.log_every==0 or it==args.epochs-1:
            wp,wd,wb=model.weights()
            row=dict(step=it+1,phase='adam',loss_pde=float(lp.detach()),
                     loss_data=float(ld.detach()),loss_bc=float(lb.detach()),
                     raw_sum=float((lp+ld+lb).detach()),objective=float(objective.detach()),
                     nu_hat=float(model.nu_hat.detach()),w_pde=float(wp),w_data=float(wd),w_bc=float(wb))
            history.append(row)
            print(f'{it+1:>7}/{args.epochs} raw_loss={row["raw_sum"]:.5e} '
                  f'lp={row["loss_pde"]:.4g} ld={row["loss_data"]:.4g} lb={row["loss_bc"]:.4g} '
                  f'nu={row["nu_hat"]:.7g}',flush=True)
        if args.save_every>0 and (it+1)%args.save_every==0:
            # Safe intermediate state for long-running 80k-step experiments.
            torch.save({'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'completed_adam_steps':it+1,'config':vars(args)},
                       output/f'checkpoint_adam_{it+1}.pt')
            pd.DataFrame(history).to_csv(output/'loss_history_partial.csv',index=False)
            print(f'Saved Adam checkpoint at step {it+1}',flush=True)
    if device.type=='cuda':torch.cuda.synchronize()
    adam_seconds=time.perf_counter()-t0
    lbfgs_seconds=0.0
    if args.lbfgs_iters>0:
        # Fixed batch is essential for L-BFGS line search/closure consistency.
        # If batch_f or batch_b =0 this uses ALL collocation and boundary points.
        lb_rng=np.random.default_rng(args.batch_seed+112233)
        ob,ph,wall=get_training_batches(obs,phys,walls,args,lb_rng)
        lbfgs=torch.optim.LBFGS([p for p in model.parameters() if p.requires_grad],
                 lr=args.lbfgs_lr,max_iter=args.lbfgs_iters,max_eval=args.lbfgs_iters*2,
                 tolerance_grad=1e-11,tolerance_change=1e-11,history_size=100,line_search_fn='strong_wolfe')
        closure_idx=[0]
        t1=time.perf_counter()
        def closure():
            lbfgs.zero_grad(set_to_none=True)
            lp=physics.pde_loss(ph,model)
            ld=physics.data_loss(model,ob)
            lb=physics.boundary_loss(model,wall)
            loss=model.weighted_loss(lp,ld,lb)
            if not torch.isfinite(loss):raise FloatingPointError('LBFGS nonfinite objective')
            loss.backward()
            closure_idx[0]+=1
            if closure_idx[0]==1 or closure_idx[0]%args.log_every==0:
                wp,wd,wb=model.weights()
                history.append(dict(step=args.epochs+closure_idx[0],phase='lbfgs_eval',
                    loss_pde=float(lp.detach()),loss_data=float(ld.detach()),loss_bc=float(lb.detach()),
                    raw_sum=float((lp+ld+lb).detach()),objective=float(loss.detach()),
                    nu_hat=float(model.nu_hat.detach()),w_pde=float(wp),w_data=float(wd),w_bc=float(wb)))
            return loss
        lbfgs.step(closure)
        if device.type=='cuda':torch.cuda.synchronize()
        lbfgs_seconds=time.perf_counter()-t1
        if model.is_adaptive:
            with torch.no_grad():model.logvars.clamp_(-args.logvar_limit,args.logvar_limit)
    train_seconds=adam_seconds+lbfgs_seconds
    metrics=export_results(output,model,df,obs,phys,walls,args,history,train_seconds,
                           adam_seconds,lbfgs_seconds,device)
    print('Saved to:',output)
    return metrics


def build_parser():
    p=argparse.ArgumentParser(description='Original PINN-MRT source matched MLP baselines (one config/seed per run)')
    p.add_argument('--baseline',choices=['plain','fourier','adaptive','fourier_adaptive'],default='fourier_adaptive',
                   help='plain=fixed-weight MLP-MRT, fourier=Fourier MLP-MRT, adaptive=learned loss weights')
    p.add_argument('--physics-mode',choices=['source','consistent'],default='source')
    p.add_argument('--data',default='num-1000.csv')
    p.add_argument('--output',default='')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--sample-seed',type=int,default=2026)
    p.add_argument('--batch-seed',type=int,default=24680)
    p.add_argument('--nu-init',type=float,default=0.005,
                   help='Common initial viscosity; original PINN source defaults to 0.001 (the known reference). Use the SAME value in KAN and all baselines.')
    p.add_argument('--true-nu',type=float,default=0.001,help='ONLY evaluated after training')
    p.add_argument('--hidden',type=int,default=60)
    p.add_argument('--init-method',choices=['source','xavier'],default='source',
                   help='source=original PINN-MRT nn.Linear default initialization; xavier=explicit Xavier normal')
    p.add_argument('--num-hidden',type=int,default=3)
    p.add_argument('--fourier-dim',type=int,default=16,help='Number of B frequency vectors')
    p.add_argument('--fourier-scale',type=float,default=1.0)
    p.add_argument('--fourier-seed',type=int,default=314159)
    p.add_argument('--include-raw',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--match-params',action='store_true',help='Match Fourier-MLP network parameter count to plain MLP as closely as possible')
    p.add_argument('--n-obs',type=int,default=1000)
    p.add_argument('--n-f',type=int,default=20000)
    p.add_argument('--n-b',type=int,default=5000,help='Boundary points PER wall')
    p.add_argument('--batch-data',type=int,default=1000,help='0=use all observations per optimizer step')
    p.add_argument('--batch-f',type=int,default=0,help='0=all 20000 collocation points, matches source full batch; smaller if OOM (then retrain other models consistently)')
    p.add_argument('--batch-b',type=int,default=0,help='0=all 5000 points PER boundary, source full batch; smaller if OOM')
    p.add_argument('--epochs',type=int,default=80000)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--nu-lr',type=float,default=1e-3,help='Original PINN-MRT uses 1e-3 for nu; set 5e-4 to match the KAN-MRT source')
    p.add_argument('--weight-lr',type=float,default=1e-3)
    p.add_argument('--logvar-limit',type=float,default=6.0)
    p.add_argument('--clip-grad',type=float,default=1.0)
    p.add_argument('--lbfgs-iters',type=int,default=100)
    p.add_argument('--lbfgs-lr',type=float,default=1e-2)
    p.add_argument('--log-every',type=int,default=1000)
    p.add_argument('--save-every',type=int,default=20000,help='Save intermediate checkpoint every N Adam steps; 0 disables')
    p.add_argument('--eval-batch',type=int,default=2048)
    return p


if __name__=='__main__':
    args=build_parser().parse_args()
    run(args)
