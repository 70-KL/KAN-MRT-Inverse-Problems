"""Time-dependent D2Q9 MRT-KAN for the 2-D cylinder-wake data set.

The spatial subdomain is fixed to [1, 8] x [-2, 2].  Two KANs represent
the equilibrium macroscopic variables (rho, u, v) and the nine
non-equilibrium populations.  Training uses velocity observations, the
continuous MRT lattice-Boltzmann residual, and the three-part velocity
boundary loss described in the accompanying PINN-MRT formulation.

This file deliberately depends only on packages already used by the
existing KAN/PINN scripts: numpy, scipy, torch and matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "cylinder_nektar_wake.mat"
DOMAIN_LB = (1.0, -2.0, 0.0)
DOMAIN_UB_XY = (8.0, 2.0)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class KANLinear(nn.Module):
    """B-spline KAN layer compatible with the existing KAN script."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 15,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=nn.SiLU,
        grid_range: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
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
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h
            + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                - 0.5
            ) * self.scale_noise / self.grid_size
            coeff = self.curve2coeff(
                self.grid.T[self.spline_order : -self.spline_order], noise
            )
            self.spline_weight.copy_(coeff)
            nn.init.kaiming_uniform_(
                self.spline_scaler, a=math.sqrt(5) * self.scale_spline
            )

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"KANLinear expected [N,{self.in_features}], got {tuple(x.shape)}"
            )
        grid = self.grid
        x3 = x.unsqueeze(-1)
        bases = ((x3 >= grid[:, :-1]) & (x3 < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left_den = (grid[:, k:-1] - grid[:, : -(k + 1)]).clamp_min(1e-12)
            right_den = (grid[:, k + 1 :] - grid[:, 1:-k]).clamp_min(1e-12)
            bases = (
                (x3 - grid[:, : -(k + 1)]) / left_den * bases[:, :, :-1]
                + (grid[:, k + 1 :] - x3) / right_den * bases[:, :, 1:]
            )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a = self.b_splines(x).transpose(0, 1)
        b = y.transpose(0, 1)
        return torch.linalg.lstsq(a, b).solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, self.in_features)
        base = F.linear(self.base_activation(x2), self.base_weight)
        spline = F.linear(
            self.b_splines(x2).reshape(x2.shape[0], -1),
            self.scaled_spline_weight.reshape(self.out_features, -1),
        )
        return (base + spline).reshape(*shape[:-1], self.out_features)

    def regularization_loss(self) -> torch.Tensor:
        l1 = self.spline_weight.abs().mean(-1)
        activation = l1.sum()
        probability = l1 / (activation + 1e-12)
        entropy = -(probability * torch.log(probability + 1e-12)).sum()
        return activation + entropy


class KAN(nn.Module):
    def __init__(
        self,
        widths: List[int],
        lb: torch.Tensor,
        ub: torch.Tensor,
        grid_size: int,
        spline_order: int,
    ) -> None:
        super().__init__()
        self.register_buffer("lb", lb.clone())
        self.register_buffer("ub", ub.clone())
        self.layers = nn.ModuleList(
            KANLinear(a, b, grid_size=grid_size, spline_order=spline_order)
            for a, b in zip(widths[:-1], widths[1:])
        )

    def forward(self, physical_x: torch.Tensor) -> torch.Tensor:
        # KAN splines operate on [0,1], while autograd still differentiates
        # with respect to the physical (x,y,t) coordinates.
        z = (physical_x - self.lb) / (self.ub - self.lb)
        z = z.clamp(0.0, 1.0 - 1e-7)
        for i, layer in enumerate(self.layers):
            z = layer(z)
            if i + 1 < len(self.layers):
                z = torch.tanh(z)
        return z

    def regularization_loss(self) -> torch.Tensor:
        return sum(layer.regularization_loss() for layer in self.layers)


def build_d2q9(device: torch.device) -> Dict[str, torch.Tensor]:
    e = torch.tensor(
        [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
         [1, 1], [-1, 1], [-1, -1], [1, -1]],
        dtype=torch.float32,
        device=device,
    )
    w = torch.tensor(
        [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9,
         1 / 36, 1 / 36, 1 / 36, 1 / 36],
        dtype=torch.float32,
        device=device,
    )
    m = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [-4, -1, -1, -1, -1, 2, 2, 2, 2],
            [4, -2, -2, -2, -2, 1, 1, 1, 1],
            [0, 1, 0, -1, 0, 1, -1, -1, 1],
            [0, -2, 0, 2, 0, 1, -1, -1, 1],
            [0, 0, 1, 0, -1, 1, 1, -1, -1],
            [0, 0, -2, 0, 2, 1, 1, -1, -1],
            [0, 1, -1, 1, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, -1, 1, -1],
        ],
        dtype=torch.float32,
        device=device,
    )
    return {"e": e, "w": w, "m": m, "m_inv": torch.linalg.inv(m)}


class CylinderMRTKAN(nn.Module):
    """Equilibrium/non-equilibrium KAN pair and MRT operators."""

    def __init__(
        self,
        lb: torch.Tensor,
        ub: torch.Tensor,
        d2q9: Dict[str, torch.Tensor],
        hidden: int = 32,
        grid_size: int = 15,
        spline_order: int = 3,
        initial_nu: float = 0.01,
        learn_nu: bool = True,
    ) -> None:
        super().__init__()
        self.eq_net = KAN([3, hidden, hidden, 3], lb, ub, grid_size, spline_order)
        self.neq_net = KAN([3, hidden, hidden, 9], lb, ub, grid_size, spline_order)
        # Keep a one-element dimension because the MRT rate vector is built
        # with torch.cat (and for checkpoint compatibility with older code).
        log_nu = torch.tensor([math.log(initial_nu)], dtype=torch.float32)
        self.log_nu = nn.Parameter(log_nu, requires_grad=learn_nu)
        self.d2q9 = d2q9

    @property
    def nu(self) -> torch.Tensor:
        return torch.exp(self.log_nu)

    def macroscopic(self, xyt: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        raw = self.eq_net(xyt)
        rho = 1.0 + 0.1 * torch.tanh(raw[:, 0:1])
        return rho, raw[:, 1:2], raw[:, 2:3]

    def equilibrium(
        self, rho: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        e, w = self.d2q9["e"], self.d2q9["w"]
        velocity = torch.cat([u, v], dim=1)
        eu = velocity @ e.T
        speed2 = (velocity * velocity).sum(dim=1, keepdim=True)
        return w[None, :] * rho * (1.0 + 3.0 * eu + 4.5 * eu**2 - 1.5 * speed2)

    def nonequilibrium(self, xyt: torch.Tensor) -> torch.Tensor:
        raw_f = self.neq_net(xyt)
        # Enforce sum(fneq)=sum(e_x*fneq)=sum(e_y*fneq)=0 exactly by
        # zeroing conserved moments rho, jx and jy.
        moments = raw_f @ self.d2q9["m"].T
        mask = torch.ones(9, dtype=raw_f.dtype, device=raw_f.device)
        mask[[0, 3, 5]] = 0.0
        moments = moments * mask
        return moments @ self.d2q9["m_inv"].T

    def collision_row_matrix(self) -> torch.Tensor:
        # Moment order: rho,e,eps,jx,qx,jy,qy,pxx,pxy.
        # Conserved moments have zero relaxation.  Shear relaxation is
        # related to viscosity by nu = (1/s_nu - 1/2)/3.
        one = torch.ones_like(self.nu)
        s_nu = 1.0 / (3.0 * self.nu + 0.5)
        rates = torch.cat(
            [0 * one, 1.1 * one, 1.0 * one, 0 * one, 1.2 * one,
             0 * one, 1.2 * one, s_nu, s_nu]
        )
        column_matrix = (
            self.d2q9["m_inv"] @ torch.diag(rates) @ self.d2q9["m"]
        )
        return column_matrix.T

    def distributions(self, xyt: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        rho, u, v = self.macroscopic(xyt)
        feq = self.equilibrium(rho, u, v)
        fneq = self.nonequilibrium(xyt)
        return rho, u, v, feq, fneq, feq + fneq

    def regularization_loss(self) -> torch.Tensor:
        return self.eq_net.regularization_loss() + self.neq_net.regularization_loss()


def streaming_derivative(
    populations: torch.Tensor, xyt: torch.Tensor, e: torch.Tensor
) -> torch.Tensor:
    terms = []
    for i in range(9):
        derivative = torch.autograd.grad(
            populations[:, i],
            xyt,
            grad_outputs=torch.ones_like(populations[:, i]),
            create_graph=True,
            retain_graph=True,
        )[0]
        terms.append(
            derivative[:, 2]
            + e[i, 0] * derivative[:, 0]
            + e[i, 1] * derivative[:, 1]
        )
    return torch.stack(terms, dim=1)


def mrt_residual(model: CylinderMRTKAN, xyt: torch.Tensor) -> torch.Tensor:
    _, _, _, feq, fneq, f = model.distributions(xyt)
    transport = streaming_derivative(f, xyt, model.d2q9["e"])
    collision = fneq @ model.collision_row_matrix()
    return transport + collision


def physics_loss(model: CylinderMRTKAN, xyt: torch.Tensor) -> torch.Tensor:
    return torch.mean(mrt_residual(model, xyt) ** 2)


def observation_loss(
    model: CylinderMRTKAN, xyt: torch.Tensor, uv: torch.Tensor
) -> torch.Tensor:
    _, u, v = model.macroscopic(xyt)
    return F.mse_loss(torch.cat([u, v], dim=1), uv)


def boundary_loss(
    model: CylinderMRTKAN,
    xyt: torch.Tensor,
    uv: torch.Tensor,
    inward_normal: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return L_mBC, L_fBC and L_EBC for one velocity boundary.

    The inward normal follows the convention in the supplied equation:
    xi_i dot n > 0 selects unknown/incoming populations.
    """
    rho, u, v, feq, fneq, f = model.distributions(xyt)
    l_mbc = (
        F.mse_loss(rho, torch.ones_like(rho))
        + F.mse_loss(u, uv[:, 0:1])
        + F.mse_loss(v, uv[:, 1:2])
    )

    e = model.d2q9["e"]
    direction = e @ inward_normal
    # Match the paper exactly: incoming xi_i.n > 0, outgoing xi_i.n < 0.
    # Rest and boundary-tangential populations (xi_i.n == 0) belong to
    # neither sum and are therefore intentionally excluded here.
    incoming = direction > 0
    outgoing = direction < 0

    # First-order non-equilibrium boundary estimate:
    # D_i f_eq + f_neq A_row = 0.  pinv is required because conserved
    # MRT modes correctly have zero relaxation rates.
    dfeq = streaming_derivative(feq, xyt, e)
    collision_row = model.collision_row_matrix()
    fneq_star = -dfeq @ torch.linalg.pinv(collision_row)
    l_fbc = F.mse_loss(fneq[:, incoming], fneq_star[:, incoming])

    residual = streaming_derivative(f, xyt, e) + fneq @ collision_row
    l_ebc = torch.mean(residual[:, outgoing] ** 2)
    return l_mbc, l_fbc, l_ebc


@dataclass
class DataBundle:
    xyt: np.ndarray
    uv: np.ndarray
    pressure: np.ndarray
    t_min: float
    t_max: float
    boundary: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
    x_static: np.ndarray
    t_values: np.ndarray
    u_grid: np.ndarray
    v_grid: np.ndarray


def load_cylinder_data(path: Path) -> DataBundle:
    data = scipy.io.loadmat(str(path))
    required = {"X_star", "t", "U_star", "p_star"}
    missing = required.difference(data)
    if missing:
        raise KeyError(f"Data file misses variables: {sorted(missing)}")
    xy = np.asarray(data["X_star"], dtype=np.float32)
    times = np.asarray(data["t"], dtype=np.float32).reshape(-1)
    velocity = np.asarray(data["U_star"], dtype=np.float32)
    pressure_grid = np.asarray(data["p_star"], dtype=np.float32)

    if not (
        np.isclose(xy[:, 0].min(), DOMAIN_LB[0])
        and np.isclose(xy[:, 0].max(), DOMAIN_UB_XY[0])
        and np.isclose(xy[:, 1].min(), DOMAIN_LB[1])
        and np.isclose(xy[:, 1].max(), DOMAIN_UB_XY[1])
    ):
        raise ValueError(
            "Expected spatial domain [1,8]x[-2,2], got "
            f"x=[{xy[:,0].min()},{xy[:,0].max()}], "
            f"y=[{xy[:,1].min()},{xy[:,1].max()}]"
        )

    n, nt = xy.shape[0], times.size
    xx = np.repeat(xy[:, 0:1], nt, axis=1)
    yy = np.repeat(xy[:, 1:2], nt, axis=1)
    tt = np.repeat(times[None, :], n, axis=0)
    xyt = np.column_stack([xx.ravel(), yy.ravel(), tt.ravel()]).astype(np.float32)
    uv = np.column_stack(
        [velocity[:, 0, :].ravel(), velocity[:, 1, :].ravel()]
    ).astype(np.float32)
    pressure = pressure_grid.ravel()[:, None].astype(np.float32)

    # Use all data values located exactly on each subdomain side.  The
    # supplied data therefore defines the time-dependent velocity BC.
    sides = {
        "left": (np.isclose(xy[:, 0], 1.0), np.array([1.0, 0.0], np.float32)),
        "right": (np.isclose(xy[:, 0], 8.0), np.array([-1.0, 0.0], np.float32)),
        "bottom": (np.isclose(xy[:, 1], -2.0), np.array([0.0, 1.0], np.float32)),
        "top": (np.isclose(xy[:, 1], 2.0), np.array([0.0, -1.0], np.float32)),
    }
    boundary = {}
    for name, (mask, normal) in sides.items():
        side_xy = xy[mask]
        side_velocity = velocity[mask]
        ns = side_xy.shape[0]
        bx = np.repeat(side_xy[:, 0:1], nt, axis=1)
        by = np.repeat(side_xy[:, 1:2], nt, axis=1)
        bt = np.repeat(times[None, :], ns, axis=0)
        bxyt = np.column_stack([bx.ravel(), by.ravel(), bt.ravel()]).astype(np.float32)
        buv = np.column_stack(
            [side_velocity[:, 0, :].ravel(), side_velocity[:, 1, :].ravel()]
        ).astype(np.float32)
        boundary[name] = (bxyt, buv, normal)

    return DataBundle(
        xyt=xyt,
        uv=uv,
        pressure=pressure,
        t_min=float(times.min()),
        t_max=float(times.max()),
        boundary=boundary,
        x_static=xy,
        t_values=times,
        u_grid=velocity[:, 0, :],
        v_grid=velocity[:, 1, :],
    )


def numpy_batch(
    arrays: Tuple[np.ndarray, ...], size: int, rng: np.random.Generator
) -> Tuple[np.ndarray, ...]:
    count = arrays[0].shape[0]
    index = rng.choice(count, size=min(size, count), replace=False)
    return tuple(array[index] for array in arrays)


def as_tensor(array: np.ndarray, device: torch.device, grad: bool = False) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32, device=device, requires_grad=grad)


def sample_collocation(
    count: int, t_min: float, t_max: float, device: torch.device
) -> torch.Tensor:
    random = torch.rand(count, 3, dtype=torch.float32, device=device)
    lower = torch.tensor([1.0, -2.0, t_min], device=device)
    upper = torch.tensor([8.0, 2.0, t_max], device=device)
    return (lower + random * (upper - lower)).requires_grad_(True)


@dataclass
class TrainConfig:
    epochs: int = 30000
    n_observation: int = 5000
    observation_batch: int = 2000
    n_collocation: int = 4000
    boundary_batch: int = 256
    hidden: int = 32
    grid_size: int = 15
    spline_order: int = 3
    learning_rate: float = 8e-4
    nu_learning_rate: float = 1e-5
    initial_nu: float = 0.01
    learn_nu: bool = True
    w_pde: float = 0.1
    w_data: float = 10.0
    w_mbc: float = 10.0
    w_fbc: float = 1.0
    w_ebc: float = 0.1
    w_reg: float = 1e-7
    pde_warmup: int = 5000
    print_every: int = 100
    save_every: int = 5000
    seed: int = 1234


def save_checkpoint(
    model: CylinderMRTKAN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: List[Dict[str, float]],
    output: Path,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
        },
        output / f"checkpoint_{epoch:06d}.pt",
    )


def save_history(history: List[Dict[str, float]], output: Path) -> None:
    if not history:
        return
    with (output / "loss_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def save_snapshot(
    model: CylinderMRTKAN,
    data: DataBundle,
    device: torch.device,
    output: Path,
    time_index: int = -1,
) -> None:
    xy = data.x_static
    t_value = float(data.t_values[time_index])
    xyt = np.column_stack(
        [xy[:, 0], xy[:, 1], np.full(xy.shape[0], t_value, dtype=np.float32)]
    ).astype(np.float32)
    model.eval()
    with torch.no_grad():
        rho, u, v = model.macroscopic(as_tensor(xyt, device))
    prediction = np.column_stack(
        [xy, np.full(xy.shape[0], t_value),
         rho.cpu().numpy(), u.cpu().numpy(), v.cpu().numpy()]
    )
    np.savetxt(
        output / "prediction_last_time.csv",
        prediction,
        delimiter=",",
        header="x,y,t,rho,u,v",
        comments="",
    )

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    values = [
        data.u_grid[:, time_index], prediction[:, 4],
        np.abs(data.u_grid[:, time_index] - prediction[:, 4]),
        data.v_grid[:, time_index], prediction[:, 5],
        np.abs(data.v_grid[:, time_index] - prediction[:, 5]),
    ]
    titles = ["u reference", "u MRT-KAN", "|u error|",
              "v reference", "v MRT-KAN", "|v error|"]
    for axis, value, title in zip(axes.ravel(), values, titles):
        artist = axis.tricontourf(xy[:, 0], xy[:, 1], value, levels=60, cmap="jet")
        fig.colorbar(artist, ax=axis)
        axis.set(xlabel="x", ylabel="y", title=title, xlim=(1, 8), ylim=(-2, 2))
    fig.suptitle(f"Cylinder wake at t={t_value:.2f}")
    fig.savefig(output / "prediction_last_time.png", dpi=250)
    plt.close(fig)


def train(
    data: DataBundle,
    config: TrainConfig,
    device: torch.device,
    output: Path,
) -> CylinderMRTKAN:
    rng = np.random.default_rng(config.seed)
    obs_xyt, obs_uv = numpy_batch(
        (data.xyt, data.uv), min(config.n_observation, len(data.xyt)), rng
    )
    lb = torch.tensor([1.0, -2.0, data.t_min], dtype=torch.float32, device=device)
    ub = torch.tensor([8.0, 2.0, data.t_max], dtype=torch.float32, device=device)
    model = CylinderMRTKAN(
        lb,
        ub,
        build_d2q9(device),
        hidden=config.hidden,
        grid_size=config.grid_size,
        spline_order=config.spline_order,
        initial_nu=config.initial_nu,
        learn_nu=config.learn_nu,
    ).to(device)

    regular_parameters = [p for name, p in model.named_parameters() if name != "log_nu"]
    groups = [{"params": regular_parameters, "lr": config.learning_rate}]
    if config.learn_nu:
        groups.append({"params": [model.log_nu], "lr": config.nu_learning_rate})
    optimizer = torch.optim.Adam(groups)
    history: List[Dict[str, float]] = []
    started = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()
        sample_xyt, sample_uv = numpy_batch(
            (obs_xyt, obs_uv), config.observation_batch, rng
        )
        x_data = as_tensor(sample_xyt, device)
        uv_data = as_tensor(sample_uv, device)
        x_f = sample_collocation(config.n_collocation, data.t_min, data.t_max, device)

        l_data = observation_loss(model, x_data, uv_data)
        l_pde = physics_loss(model, x_f)
        l_mbc = torch.zeros((), device=device)
        l_fbc = torch.zeros((), device=device)
        l_ebc = torch.zeros((), device=device)
        for bxyt_pool, buv_pool, normal in data.boundary.values():
            bxyt, buv = numpy_batch((bxyt_pool, buv_pool), config.boundary_batch, rng)
            side_losses = boundary_loss(
                model,
                as_tensor(bxyt, device, grad=True),
                as_tensor(buv, device),
                as_tensor(normal, device),
            )
            l_mbc = l_mbc + side_losses[0]
            l_fbc = l_fbc + side_losses[1]
            l_ebc = l_ebc + side_losses[2]

        l_reg = model.regularization_loss()
        pde_ramp = min(1.0, epoch / max(config.pde_warmup, 1))
        total = (
            config.w_data * l_data
            + pde_ramp * config.w_pde * l_pde
            + config.w_mbc * l_mbc
            + config.w_fbc * l_fbc
            + config.w_ebc * l_ebc
            + config.w_reg * l_reg
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        row = {
            "epoch": epoch,
            "total": float(total.detach()),
            "data": float(l_data.detach()),
            "pde": float(l_pde.detach()),
            "mBC": float(l_mbc.detach()),
            "fBC": float(l_fbc.detach()),
            "EBC": float(l_ebc.detach()),
            "nu": float(model.nu.detach()),
        }
        history.append(row)
        if epoch == 1 or epoch % config.print_every == 0:
            elapsed = time.time() - started
            print(
                f"epoch {epoch:06d}/{config.epochs} total={row['total']:.3e} "
                f"data={row['data']:.3e} pde={row['pde']:.3e} "
                f"BC=({row['mBC']:.3e},{row['fBC']:.3e},{row['EBC']:.3e}) "
                f"nu={row['nu']:.6e} time={elapsed:.1f}s"
            )
        if config.save_every > 0 and epoch % config.save_every == 0:
            save_checkpoint(model, optimizer, epoch, history, output)
            save_history(history, output)

    save_checkpoint(model, optimizer, config.epochs, history, output)
    save_history(history, output)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--n-observation", type=int, default=5000)
    parser.add_argument("--observation-batch", type=int, default=2000)
    parser.add_argument("--n-collocation", type=int, default=4000)
    parser.add_argument("--boundary-batch", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--spline-order", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--nu-learning-rate", type=float, default=1e-5)
    parser.add_argument("--initial-nu", type=float, default=0.01)
    parser.add_argument("--fixed-nu", action="store_true")
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="one very small epoch for installation/code verification",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"Using device: {device}")
    print(f"Loading data: {args.data.resolve()}")
    data = load_cylinder_data(args.data.resolve())

    config = TrainConfig(
        epochs=args.epochs,
        n_observation=args.n_observation,
        observation_batch=args.observation_batch,
        n_collocation=args.n_collocation,
        boundary_batch=args.boundary_batch,
        hidden=args.hidden,
        grid_size=args.grid_size,
        spline_order=args.spline_order,
        learning_rate=args.learning_rate,
        nu_learning_rate=args.nu_learning_rate,
        initial_nu=args.initial_nu,
        learn_nu=not args.fixed_nu,
        print_every=args.print_every,
        save_every=args.save_every,
        seed=args.seed,
    )
    if args.smoke_test:
        config.epochs = 1
        config.n_observation = 32
        config.observation_batch = 16
        config.n_collocation = 8
        config.boundary_batch = 4
        config.hidden = 8
        config.grid_size = 5
        config.print_every = 1
        config.save_every = 0

    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = SCRIPT_DIR / "results" / f"cylinder_kan_mrt_{stamp}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, ensure_ascii=False)
    print(f"Results: {output}")

    model = train(data, config, device, output)
    save_snapshot(model, data, device, output)
    print(f"Finished. Learned viscosity nu={float(model.nu.detach()):.8e}")


if __name__ == "__main__":
    main()
