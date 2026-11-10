'''
@Project ：Improved physics-informed neural network in mitigating gradient-related failure
@File    ：Lid_driven_Cavity_PINN_MLP.py
@IDE     ：PyCharm
@Author  ：Pancheng Niu (Modified by Grok)
@Date    ：2024/7/18 下午3:41
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pandas as pd
import os

# 设置随机种子
torch.manual_seed(2341)
np.random.seed(2341)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current device:", device)

# MLP网络定义
class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.layers = layers
        self.activation = nn.Tanh()
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        a = self.activation(self.linear[0](x))
        for i in range(1, len(self.layers) - 2):
            a = self.activation(self.linear[i](a))
        a = self.linear[-1](a)
        return a

# 采样器类
class Sampler:
    def __init__(self, dim, coords, func, name=None):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name

    def sample(self, N):
        x = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * np.random.rand(N, self.dim)
        y = self.func(x)
        return torch.tensor(x, requires_grad=True).float().to(device), torch.tensor(y, requires_grad=True).float().to(device)

# Navier-Stokes PINN类
class Navier_Stokes2D:
    def __init__(self, layers, bcs_sampler, res_sampler, operator, Re):
        X, _ = res_sampler.sample(np.int32(1e5))
        self.mu_X = X.mean(0).detach()
        self.sigma_X = X.std(0).detach()
        self.sigma_x = self.sigma_X[0]
        self.sigma_y = self.sigma_X[1]

        self.layers = layers
        self.dnn = Net(layers).to(device)
        self.operator = operator
        self.bcs_sampler = bcs_sampler
        self.res_sampler = res_sampler
        self.Re = torch.tensor(Re, dtype=torch.float32)

        self.optimizer = optim.Adam(self.dnn.parameters(), lr=1e-3)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.iter = 0
        self.loss_res_log = []
        self.loss_bcs_log = []
        self.loss_log = []

    def d(self, f, x):
        return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]

    def net_psi_p(self, x, y):
        psi_p = self.dnn(torch.cat([x, y], 1))
        psi = psi_p[:, 0:1]
        p = psi_p[:, 1:2]
        return psi, p

    def net_uv(self, x, y):
        psi, p = self.net_psi_p(x, y)
        u = self.d(psi, y) / self.sigma_y
        v = -self.d(psi, x) / self.sigma_x
        return u, v

    def net_r(self, x, y):
        psi, p = self.net_psi_p(x, y)
        u_momentum_pred, v_momentum_pred = self.operator(psi, p, x, y, self.Re, self.sigma_x, self.sigma_y)
        return u_momentum_pred, v_momentum_pred

    def fetch_minibatch(self, sampler, batch_size):
        X, Y = sampler.sample(batch_size)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    def train(self, nIter=40000, batch_size=128):
        self.dnn.train()
        for it in range(nIter):
            X_bc1_batch, u_bc1_batch = self.fetch_minibatch(self.bcs_sampler[0], batch_size)
            X_bc2_batch, u_bc2_batch = self.fetch_minibatch(self.bcs_sampler[1], batch_size)
            X_bc3_batch, u_bc3_batch = self.fetch_minibatch(self.bcs_sampler[2], batch_size)
            X_bc4_batch, u_bc4_batch = self.fetch_minibatch(self.bcs_sampler[3], batch_size)
            X_res_batch, f_res_batch = self.fetch_minibatch(self.res_sampler, batch_size)

            u_bc1_pred, v_bc1_pred = self.net_uv(X_bc1_batch[:, 0:1], X_bc1_batch[:, 1:2])
            u_bc2_pred, v_bc2_pred = self.net_uv(X_bc2_batch[:, 0:1], X_bc2_batch[:, 1:2])
            u_bc3_pred, v_bc3_pred = self.net_uv(X_bc3_batch[:, 0:1], X_bc3_batch[:, 1:2])
            u_bc4_pred, v_bc4_pred = self.net_uv(X_bc4_batch[:, 0:1], X_bc4_batch[:, 1:2])

            U_bc1_pred = torch.cat((u_bc1_pred, v_bc1_pred), dim=1)
            U_bc2_pred = torch.cat((u_bc2_pred, v_bc2_pred), dim=1)
            U_bc3_pred = torch.cat((u_bc3_pred, v_bc3_pred), dim=1)
            U_bc4_pred = torch.cat((u_bc4_pred, v_bc4_pred), dim=1)

            u_momentum_pred, v_momentum_pred = self.net_r(X_res_batch[:, 0:1], X_res_batch[:, 1:2])

            loss_res = torch.mean(torch.square(u_momentum_pred)) + torch.mean(torch.square(v_momentum_pred))
            loss_bcs = torch.mean(torch.square(U_bc1_pred - u_bc1_batch) +
                                  torch.square(U_bc2_pred) +
                                  torch.square(U_bc3_pred) +
                                  torch.square(U_bc4_pred))
            loss = loss_res + loss_bcs

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 记录损失
            self.loss_res_log.append(loss_res.item())
            self.loss_bcs_log.append(loss_bcs.item())
            self.loss_log.append(loss.item())

            if it % 1000 == 0:
                self.scheduler.step()
            if it % 100 == 0:
                print(f'Iter {it}, Loss: {loss.item():.3e}, Loss_res: {loss_res.item():.3e}, Loss_bcs: {loss_bcs.item():.3e}')

    def predict_psi_p(self, X_star):
        X_star = torch.tensor(X_star, requires_grad=True).float().to(device)
        X_star = (X_star - self.mu_X) / self.sigma_X
        psi_star, p_star = self.net_psi_p(X_star[:, 0:1], X_star[:, 1:2])
        return psi_star, p_star

    def predict_uv(self, X_star):
        X_star = torch.tensor(X_star, requires_grad=True).float().to(device)
        X_star = (X_star - self.mu_X) / self.sigma_X
        u_star, v_star = self.net_uv(X_star[:, 0:1], X_star[:, 1:2])
        return u_star, v_star

# 主函数
if __name__ == '__main__':
    # 定义边界条件和源项
    def U_gamma_1(x):
        num = x.shape[0]
        return np.tile(np.array([1.0, 0.0]), (num, 1))

    def U_gamma_2(x):
        num = x.shape[0]
        return np.zeros((num, 2))

    def f(x):
        num = x.shape[0]
        return np.zeros((num, 2))

    def d(f, x):
        return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]

    def operator(psi, p, x, y, Re, sigma_x=1.0, sigma_y=1.0):
        u = d(psi, y) / sigma_y
        v = -d(psi, x) / sigma_x
        u_x = d(u, x) / sigma_x
        u_y = d(u, y) / sigma_y
        v_x = d(v, x) / sigma_x
        v_y = d(v, y) / sigma_y
        p_x = d(p, x) / sigma_x
        p_y = d(p, y) / sigma_y
        u_xx = d(u_x, x) / sigma_x
        u_yy = d(u_y, y) / sigma_y
        v_xx = d(v_x, x) / sigma_x
        v_yy = d(v_y, y) / sigma_y
        Ru_momentum = u * u_x + v * u_y + p_x - (u_xx + u_yy) / Re
        Rv_momentum = u * v_x + v * v_y + p_y - (v_xx + v_yy) / Re
        return Ru_momentum, Rv_momentum

    Re = 100.0

    # 定义边界和域坐标
    bc1_coords = np.array([[0.0, 1.0], [1.0, 1.0]])
    bc2_coords = np.array([[0.0, 0.0], [0.0, 1.0]])
    bc3_coords = np.array([[1.0, 0.0], [1.0, 1.0]])
    bc4_coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    dom_coords = np.array([[0.0, 0.0], [1.0, 1.0]])

    bc1 = Sampler(2, bc1_coords, lambda x: U_gamma_1(x), name='Dirichlet BC1')
    bc2 = Sampler(2, bc2_coords, lambda x: U_gamma_2(x), name='Dirichlet BC2')
    bc3 = Sampler(2, bc3_coords, lambda x: U_gamma_2(x), name='Dirichlet BC3')
    bc4 = Sampler(2, bc4_coords, lambda x: U_gamma_2(x), name='Dirichlet BC4')
    bcs_sampler = [bc1, bc2, bc3, bc4]
    res_sampler = Sampler(2, dom_coords, lambda x: f(x), name='Forcing')

    # 设置网络结构并训练
    # layers = [2] + [30] * 5 + [2]
    layers = [2] + [30] * 5 + [2]
    model = Navier_Stokes2D(layers, bcs_sampler, res_sampler, operator, Re)
    model.train(nIter=40000, batch_size=128)

    # 生成预测网格
    nx, ny = 100, 100
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)
    X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

    # 预测结果
    psi_pred, p_pred = model.predict_psi_p(X_star)
    u_pred, v_pred = model.predict_uv(X_star)

    u_pred = u_pred.detach().cpu().numpy()
    v_pred = v_pred.detach().cpu().numpy()
    psi_pred = psi_pred.detach().cpu().numpy()
    p_pred = p_pred.detach().cpu().numpy()

    u_star = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
    v_star = griddata(X_star, v_pred.flatten(), (X, Y), method='cubic')
    velocity_star = griddata(X_star, np.sqrt(u_pred**2 + v_pred**2).flatten(), (X, Y), method='cubic')

    # 参考解（假设已有）
    u_ref = np.genfromtxt("reference_u.csv", delimiter=',')
    v_ref = np.genfromtxt("reference_v.csv", delimiter=',')

    # 绘制速度 u 和 v 的结果图（包含绝对误差）
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    im1 = axs[0, 0].imshow(u_ref.T, extent=(0, 1, 0, 1), origin='lower', cmap='jet')
    plt.colorbar(im1, ax=axs[0, 0])
    axs[0, 0].set_title('Reference u')
    im2 = axs[0, 1].imshow(u_star, extent=(0, 1, 0, 1), origin='lower', cmap='jet')
    plt.colorbar(im2, ax=axs[0, 1])
    axs[0, 1].set_title('Predicted u')
    im3 = axs[0, 2].imshow(np.abs(u_star - u_ref.T), extent=(0, 1, 0, 1), origin='lower', cmap='jet')
    plt.colorbar(im3, ax=axs[0, 2])
    axs[0, 2].set_title('Absolute Error u')
    im4 = axs[1, 0].imshow(v_ref.T, extent=(0, 1, 0, 1), origin='lower', cmap='jet')
    plt.colorbar(im4, ax=axs[1, 0])
    axs[1, 0].set_title('Reference v')
    im5 = axs[1, 1].imshow(v_star, extent=(0, 1, 0, 1), origin='lower', cmap='jet')
    plt.colorbar(im5, ax=axs[1, 1])
    axs[1, 1].set_title('Predicted v')
    im6 = axs[1, 2].imshow(np.abs(v_star - v_ref.T), extent=(0, 1, 0, 1), origin='lower', cmap='jet')
    plt.colorbar(im6, ax=axs[1, 2])
    axs[1, 2].set_title('Absolute Error v')
    plt.tight_layout()
    plt.savefig('PINN_uv_results.pdf', dpi=800, bbox_inches='tight')
    plt.close()

    # 沿中心线的速度剖面
    x_center = 0.5 * np.ones_like(y)
    y_center = 0.5 * np.ones_like(x)
    X_xcenter = np.hstack((x_center[:, None], y[:, None]))
    X_ycenter = np.hstack((x[:, None], y_center[:, None]))

    u_xcenter, v_xcenter = model.predict_uv(X_xcenter)
    u_ycenter, v_ycenter = model.predict_uv(X_ycenter)

    u_xcenter = u_xcenter.detach().cpu().numpy()
    v_xcenter = v_xcenter.detach().cpu().numpy()
    u_ycenter = u_ycenter.detach().cpu().numpy()
    v_ycenter = v_ycenter.detach().cpu().numpy()

    # 绘制中心线速度剖面
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(y, u_xcenter, label='Predicted u (x=0.5)')
    ax1.plot(y, u_ref[:, 50], label='Reference u (x=0.5)')
    ax1.set_xlabel('y')
    ax1.set_ylabel('u')
    ax1.legend()
    ax2.plot(x, v_ycenter, label='Predicted v (y=0.5)')
    ax2.plot(x, v_ref[50, :], label='Reference v (y=0.5)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('v')
    ax2.legend()
    plt.tight_layout()
    plt.savefig('PINN_centerline_velocity.pdf', dpi=800, bbox_inches='tight')
    plt.close()

    # 保存数据到Excel
    os.makedirs('results', exist_ok=True)
    centerline_data = {
        'y': y,
        'u_xcenter_pred': u_xcenter.flatten(),
        'u_xcenter_ref': u_ref[:, 50],
        'x': x,
        'v_ycenter_pred': v_ycenter.flatten(),
        'v_ycenter_ref': v_ref[50, :]
    }
    df = pd.DataFrame(centerline_data)
    df.to_excel('results/PINN_centerline_velocity.xlsx', index=False)

    # 绘制速度场图
    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(np.sqrt(u_ref**2 + v_ref**2).T, extent=(0, 1, 0, 1), origin='lower', cmap='jet')
    plt.colorbar(im1, ax=ax1)
    ax1.set_title('Reference Velocity')
    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(velocity_star, extent=(0, 1, 0, 1), origin='lower', cmap='jet')
    plt.colorbar(im2, ax=ax2)
    ax2.set_title('Predicted Velocity')
    ax3 = fig.add_subplot(133)
    im3 = ax3.imshow(np.abs(velocity_star - np.sqrt(u_ref**2 + v_ref**2).T), extent=(0, 1, 0, 1), origin='lower', cmap='jet')
    plt.colorbar(im3, ax=ax3)
    ax3.set_title('Absolute Error')
    plt.tight_layout()
    plt.savefig('PINN_velocity_results.pdf', dpi=800, bbox_inches='tight')
    plt.close()

    # 绘制损失曲线
    fig = plt.figure(figsize=(10, 6))
    plt.plot(model.loss_log, label='Total Loss')
    plt.plot(model.loss_res_log, label='Residual Loss')
    plt.plot(model.loss_bcs_log, label='Boundary Loss')
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig('PINN_loss.pdf', dpi=800, bbox_inches='tight')
    plt.close()

    # 计算并打印相对L2误差
    def relative_l2_error(pred, ref):
        error = np.sqrt(np.mean((pred - ref) ** 2))
        norm_ref = np.sqrt(np.mean(ref ** 2))
        return error / norm_ref if norm_ref > 1e-10 else float('inf')

    # 计算 u、v 和总速度场的相对L2误差
    u_l2_error = relative_l2_error(u_star, u_ref.T)
    v_l2_error = relative_l2_error(v_star, v_ref.T)
    velocity_ref = np.sqrt(u_ref**2 + v_ref**2).T
    velocity_l2_error = relative_l2_error(velocity_star, velocity_ref)

    # 打印误差
    print(f'Relative L2 Error for u: {u_l2_error:.6e}')
    print(f'Relative L2 Error for v: {v_l2_error:.6e}')
    print(f'Relative L2 Error for velocity: {velocity_l2_error:.6e}')