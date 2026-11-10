'''
@Project ：Improved physics-informed neural network in mitigating gradient-related failure
@File    ：Lid_driven_Cavity_PINN_MLP_Inverse.py
@IDE     ：PyCharm
@Author  ：Adapted for inverse problem with user data
@Date    ：2024/7/20
'''
import torch
import torch.nn as nn
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import timeit
import random
from tqdm import trange
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pandas as pd
import os


def load_data(file_path, num_points=None):
    """加载观测数据并采样"""
    try:
        data = pd.read_csv(file_path)
        required_columns = ['X', 'Y', 'u', 'v']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"数据文件中缺少列: {col}")
        x_data = data['X'].values[:, None]
        y_data = data['Y'].values[:, None]
        u_data = data['u'].values[:, None]
        v_data = data['v'].values[:, None]

        # 如果指定了采样点数，随机选择部分数据
        if num_points is not None and num_points < len(x_data):
            indices = np.random.choice(len(x_data), num_points, replace=False)
            x_data = x_data[indices]
            y_data = y_data[indices]
            u_data = u_data[indices]
            v_data = v_data[indices]

        return x_data, y_data, u_data, v_data
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到数据文件: {file_path}")


# Define the pure MLP network (Net)
class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.layers = layers
        self.iter = 0
        self.activation = nn.Tanh()
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)

        # Additional output for the unknown parameter (log(nu))
        self.log_nu = nn.Parameter(torch.tensor([-4.6052], requires_grad=True))  # Initial guess for nu=0.01 (Re=100)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        a = self.activation(self.linear[0](x))
        for i in range(1, len(self.layers) - 2):
            z = self.linear[i](a)
            a = self.activation(z)
        a = self.linear[-1](a)
        return a, torch.exp(self.log_nu)  # Return both network output and predicted nu


# Set random seed for reproducibility
def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


seed_torch(42)

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current device:", device)


# Sampler class for boundary, residual and data points
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


# Navier-Stokes solver using PINN with MLP for inverse problem
class Navier_Stokes2D_Inverse:
    def __init__(self, layers, bcs_sampler, res_sampler, data_sampler, operator, true_nu=None):
        X, _ = res_sampler.sample(np.int32(1e5))
        self.mu_X = X.mean(0).detach()
        self.sigma_X = X.std(0).detach()
        self.sigma_x = self.sigma_X[0]
        self.sigma_y = self.sigma_X[1]

        self.layers = layers
        self.operator = operator
        self.bcs_sampler = bcs_sampler
        self.res_sampler = res_sampler
        self.data_sampler = data_sampler
        self.true_nu = true_nu  # True nu for monitoring during training (if known)

        # Initialize MLP network
        self.dnn = Net(layers).to(device)

        self.optimizer_1 = torch.optim.Adam(self.dnn.parameters(), lr=1e-3)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer_1, gamma=0.9)
        self.iter = 0

        self.loss_bcs_log = []
        self.loss_res_log = []
        self.loss_data_log = []
        self.loss_log = []
        self.nu_pred_log = []

    def d(self, f, x):
        return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]

    def net_psi_p(self, x, y):
        psi_p, _ = self.dnn(torch.cat([x, y], 1))
        psi = psi_p[:, 0:1]
        p = psi_p[:, 1:2]
        return psi, p

    def net_uv(self, x, y):
        psi, p = self.net_psi_p(x, y)
        u = self.d(psi, y) / self.sigma_y
        v = -self.d(psi, x) / self.sigma_x
        return u, v

    def net_r(self, x, y, nu):
        psi, p = self.net_psi_p(x, y)
        u_momentum_pred, v_momentum_pred = self.operator(psi, p, x, y, nu, self.sigma_x, self.sigma_y)
        return u_momentum_pred, v_momentum_pred

    def fetch_minibatch(self, sampler, batch_size):
        X, Y = sampler.sample(batch_size)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    def epoch_train(self, batch_size):
        X_bc1_batch, u_bc1_batch = self.fetch_minibatch(self.bcs_sampler[0], batch_size)
        X_bc2_batch, u_bc2_batch = self.fetch_minibatch(self.bcs_sampler[1], batch_size)
        X_bc3_batch, u_bc3_batch = self.fetch_minibatch(self.bcs_sampler[2], batch_size)
        X_bc4_batch, u_bc4_batch = self.fetch_minibatch(self.bcs_sampler[3], batch_size)
        X_res_batch, f_res_batch = self.fetch_minibatch(self.res_sampler, batch_size)
        X_data_batch, u_data_batch = self.fetch_minibatch(self.data_sampler, batch_size)

        # Get current predicted nu
        _, nu_pred = self.dnn(torch.zeros(1, 2).to(device))

        # Boundary conditions loss
        u_bc1_pred, v_bc1_pred = self.net_uv(X_bc1_batch[:, 0:1], X_bc1_batch[:, 1:2])
        u_bc2_pred, v_bc2_pred = self.net_uv(X_bc2_batch[:, 0:1], X_bc2_batch[:, 1:2])
        u_bc3_pred, v_bc3_pred = self.net_uv(X_bc3_batch[:, 0:1], X_bc3_batch[:, 1:2])
        u_bc4_pred, v_bc4_pred = self.net_uv(X_bc4_batch[:, 0:1], X_bc4_batch[:, 1:2])

        U_bc1_pred = torch.cat((u_bc1_pred, v_bc1_pred), dim=1)
        U_bc2_pred = torch.cat((u_bc2_pred, v_bc2_pred), dim=1)
        U_bc3_pred = torch.cat((u_bc3_pred, v_bc3_pred), dim=1)
        U_bc4_pred = torch.cat((u_bc4_pred, v_bc4_pred), dim=1)

        loss_bcs = torch.mean(torch.square(U_bc1_pred - u_bc1_batch) +
                              torch.square(U_bc2_pred) +
                              torch.square(U_bc3_pred) +
                              torch.square(U_bc4_pred))

        # PDE residual loss
        u_momentum_pred, v_momentum_pred = self.net_r(X_res_batch[:, 0:1], X_res_batch[:, 1:2], nu_pred)
        loss_u_momentum = torch.mean(torch.square(u_momentum_pred))
        loss_v_momentum = torch.mean(torch.square(v_momentum_pred))
        loss_res = loss_u_momentum + loss_v_momentum

        # Data loss
        u_data_pred, v_data_pred = self.net_uv(X_data_batch[:, 0:1], X_data_batch[:, 1:2])
        U_data_pred = torch.cat((u_data_pred, v_data_pred), dim=1)
        loss_data = torch.mean(torch.square(U_data_pred - u_data_batch))

        return loss_res, loss_bcs, loss_data, nu_pred

    def loss_fun(self, loss_bcs, loss_res, loss_data):
        loss = loss_bcs + loss_res + loss_data
        return loss

    def train(self, nIter=10000, batch_size=128):
        start_time = timeit.default_timer()
        print(f"Model: PINN Inverse, Layer: {self.layers}")
        self.dnn.train()
        pbar = trange(nIter, ncols=140)
        for it in pbar:
            loss_res, loss_bcs, loss_data, nu_pred = self.epoch_train(batch_size)
            self.optimizer_1.zero_grad()
            loss = self.loss_fun(loss_bcs, loss_res, loss_data)
            loss.backward()
            self.optimizer_1.step()

            true_loss = loss_res + loss_bcs + loss_data
            self.loss_bcs_log.append(loss_bcs.item())
            self.loss_res_log.append(loss_res.item())
            self.loss_data_log.append(loss_data.item())
            self.loss_log.append(true_loss.item())
            self.nu_pred_log.append(nu_pred.item())

            if self.iter % 1000 == 0:
                self.scheduler.step()
            if it % 100 == 0:
                pbar.set_postfix({'Iter': self.iter,
                                  'Loss': '{0:.3e}'.format(true_loss.item()),
                                  'nu_pred': '{0:.6f}'.format(nu_pred.item()),
                                  'true_nu': '{0:.6f}'.format(self.true_nu) if self.true_nu is not None else 'N/A'})
            self.iter += 1
        elapsed = timeit.default_timer() - start_time
        print("Time: {0:.2f}s".format(elapsed))
        print(f"Final predicted nu: {nu_pred.item():.6f}")
        if self.true_nu is not None:
            print(f"True nu: {self.true_nu:.6f}")
            print(f"Relative error: {abs(nu_pred.item() - self.true_nu) / self.true_nu * 100:.2f}%")

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

    def predict_nu(self):
        with torch.no_grad():
            _, nu = self.dnn(torch.zeros(1, 2).to(device))
        return nu.item()


# Differential operator for Navier-Stokes
def d(f, x):
    return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]


def operator(psi, p, x, y, nu, sigma_x=1.0, sigma_y=1.0):
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

    Ru_momentum = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    Rv_momentum = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    return Ru_momentum, Rv_momentum


# Main execution block
if __name__ == '__main__':
    # Problem parameters
    true_Re = 100.0  # True Reynolds number
    true_nu = 1.0 / true_Re  # True viscosity

    # 创建保存目录
    output_dir = 'prediction_results'
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"创建目录失败: {e}")
            exit(1)

    # 加载数据
    data_file = 'u-v-p-100.csv'  # 确保文件为CSV格式
    try:
        x_data, y_data, u_data, v_data = load_data(data_file, num_points=1000)
        print(f"成功加载数据: {data_file}, 数据点数: {len(x_data)}")
    except (FileNotFoundError, ValueError) as e:
        print(f"加载数据失败: {e}")
        exit(1)

    # Boundary condition functions
    def U_gamma_1(x):
        num = x.shape[0]
        return np.tile(np.array([1.0, 0.0]), (num, 1))

    def U_gamma_2(x):
        num = x.shape[0]
        return np.zeros((num, 2))

    def f(x):
        num = x.shape[0]
        return np.zeros((num, 2))

    # Define domain and boundary conditions
    bc1_coords = np.array([[0.0, 1.0], [1.0, 1.0]])  # Top wall (moving lid)
    bc2_coords = np.array([[0.0, 0.0], [0.0, 1.0]])  # Left wall
    bc3_coords = np.array([[1.0, 0.0], [1.0, 1.0]])  # Right wall
    bc4_coords = np.array([[0.0, 0.0], [1.0, 0.0]])  # Bottom wall
    dom_coords = np.array([[0.0, 0.0], [1.0, 1.0]])  # Domain

    bc1 = Sampler(2, bc1_coords, lambda x: U_gamma_1(x), name='Dirichlet BC1')
    bc2 = Sampler(2, bc2_coords, lambda x: U_gamma_2(x), name='Dirichlet BC2')
    bc3 = Sampler(2, bc3_coords, lambda x: U_gamma_2(x), name='Dirichlet BC3')
    bc4 = Sampler(2, bc4_coords, lambda x: U_gamma_2(x), name='Dirichlet BC4')
    bcs_sampler = [bc1, bc2, bc3, bc4]

    res_sampler = Sampler(2, dom_coords, lambda x: f(x), name='Forcing')

    # 组合x_data和y_data为X_data，u_data和v_data为U_data
    X_data = np.hstack((x_data, y_data))
    U_data = np.hstack((u_data, v_data))

    # Create data sampler using your data
    def data_func(x):
        # Find the closest points in X_data
        indices = np.argmin(np.sum((x[:, None, :] - X_data[None, :, :])**2, axis=2), axis=1)
        return U_data[indices]

    data_sampler = Sampler(2, dom_coords, data_func, name='Data')

    # Now solve the inverse problem
    print("\nSolving inverse problem with your data...")
    layer = [2] + [60] * 5 + [2]  # 5 hidden layers with 60 neurons each (修正注释)
    inverse_model = Navier_Stokes2D_Inverse(layer, bcs_sampler, res_sampler, data_sampler, operator, true_nu)
    inverse_model.train(nIter=40000, batch_size=128)

    # ============== 保存最终预测的nu值 ==============
    final_nu = inverse_model.predict_nu()
    with open(f'{output_dir}/predicted_nu.txt', 'w') as f:
        f.write(f"Predicted viscosity (nu): {final_nu:.8f}\n")
        f.write(f"Predicted Reynolds number (Re): {1/final_nu:.2f}\n")
        if true_nu is not None:
            rel_error = abs(final_nu-true_nu)/true_nu*100
            f.write(f"True viscosity (nu): {true_nu:.8f}\n")
            f.write(f"Relative error: {rel_error:.2f}%\n")
    print(f"最终预测的粘度值已保存至: {output_dir}/predicted_nu.txt")

    # ============== 保存训练损失数据 ==============
    np.savetxt(f'{output_dir}/loss_residual.csv', inverse_model.loss_res_log, delimiter=',')
    np.savetxt(f'{output_dir}/loss_boundary.csv', inverse_model.loss_bcs_log, delimiter=',')
    np.savetxt(f'{output_dir}/loss_data.csv', inverse_model.loss_data_log, delimiter=',')
    np.savetxt(f'{output_dir}/loss_total.csv', inverse_model.loss_log, delimiter=',')
    np.savetxt(f'{output_dir}/predicted_nu_history.csv', inverse_model.nu_pred_log, delimiter=',')
    print(f"训练损失数据已保存至: {output_dir}")

    # ============== 预测和可视化 ==============
    N = 100  # 网格分辨率
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y)
    X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

    # 预测所有场变量
    print("正在进行预测...")
    psi_pred, p_pred = inverse_model.predict_psi_p(X_star)
    u_pred, v_pred = inverse_model.predict_uv(X_star)

    # 转换为numpy数组并reshape
    u_pred = u_pred.detach().cpu().numpy().reshape(N, N)
    v_pred = v_pred.detach().cpu().numpy().reshape(N, N)
    p_pred = p_pred.detach().cpu().numpy().reshape(N, N)
    psi_pred = psi_pred.detach().cpu().numpy().reshape(N, N)

    # 计算流函数(ψ) - 通过积分速度场
    psi_calc = np.zeros((N, N))
    for i in range(1, N):
        psi_calc[i, :] = psi_calc[i - 1, :] + v_pred[i, :] * (y[1] - y[0])
    for j in range(1, N):
        psi_calc[:, j] = psi_calc[:, j - 1] - u_pred[:, j] * (x[1] - x[0])

    # 统一绘图参数
    plt_params = {
        'cmap': 'jet',
        'levels': 50,
        'figsize': (6, 5),
        'dpi': 300,
        'bbox_inches': 'tight'
    }

    # 1. 保存u速度场
    plt.figure(figsize=plt_params['figsize'])
    contour = plt.contourf(X, Y, u_pred, levels=plt_params['levels'], cmap=plt_params['cmap'])
    plt.colorbar(contour, label='u velocity')
    plt.title('PINN Inverse - u velocity')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'{output_dir}/PINN_inverse_u_velocity.png', dpi=plt_params['dpi'], bbox_inches=plt_params['bbox_inches'])
    plt.close()

    # 2. 保存v速度场
    plt.figure(figsize=plt_params['figsize'])
    contour = plt.contourf(X, Y, v_pred, levels=plt_params['levels'], cmap=plt_params['cmap'])
    plt.colorbar(contour, label='v velocity')
    plt.title('PINN Inverse - v velocity')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'{output_dir}/PINN_inverse_v_velocity.png', dpi=plt_params['dpi'], bbox_inches=plt_params['bbox_inches'])
    plt.close()

    # 3. 保存压力场(p)
    plt.figure(figsize=plt_params['figsize'])
    contour = plt.contourf(X, Y, p_pred, levels=plt_params['levels'], cmap=plt_params['cmap'])
    plt.colorbar(contour, label='Pressure (p)')
    plt.title('PINN Inverse - Pressure')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'{output_dir}/PINN_inverse_pressure.png', dpi=plt_params['dpi'], bbox_inches=plt_params['bbox_inches'])
    plt.close()

    # 4. 保存流函数(ψ)
    plt.figure(figsize=plt_params['figsize'])
    contour = plt.contourf(X, Y, psi_calc, levels=plt_params['levels'], cmap=plt_params['cmap'])
    plt.colorbar(contour, label='Stream function (ψ)')
    plt.title('PINN Inverse - Stream Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'{output_dir}/PINN_inverse_stream_function.png', dpi=plt_params['dpi'], bbox_inches=plt_params['bbox_inches'])
    plt.close()

    # 5. 保存速度幅值
    velocity_magnitude = np.sqrt(u_pred ** 2 + v_pred ** 2)
    plt.figure(figsize=plt_params['figsize'])
    contour = plt.contourf(X, Y, velocity_magnitude, levels=plt_params['levels'], cmap=plt_params['cmap'])
    plt.colorbar(contour, label='Velocity magnitude')
    plt.title('PINN Inverse - Velocity Magnitude')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'{output_dir}/PINN_inverse_velocity_magnitude.png', dpi=plt_params['dpi'], bbox_inches=plt_params['bbox_inches'])
    plt.close()

    # 6. 损失曲线
    plt.figure(figsize=(8, 5))
    plt.plot(inverse_model.loss_res_log, label='Residual Loss', linewidth=1.5)
    plt.plot(inverse_model.loss_bcs_log, label='BCs Loss', linewidth=1.5)
    plt.plot(inverse_model.loss_data_log, label='Data Loss', linewidth=1.5)
    plt.plot(inverse_model.loss_log, label='Total Loss', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.title('PINN Inverse Training Loss History', fontsize=14)
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(f'{output_dir}/PINN_inverse_training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 7. 参数识别过程
    plt.figure(figsize=(8, 5))
    plt.plot(inverse_model.nu_pred_log, label='Predicted nu', linewidth=2)
    if true_nu is not None:
        plt.axhline(y=true_nu, color='r', linestyle='--', label='True nu')
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Viscosity (nu)', fontsize=12)
    plt.legend(fontsize=10)
    plt.title('Parameter Identification History', fontsize=14)
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(f'{output_dir}/PINN_inverse_parameter_identification.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 8. 显示数据点位置
    plt.figure(figsize=(6, 5))
    plt.scatter(X_data[:, 0], X_data[:, 1], s=5, alpha=0.5)
    plt.title('Data Points Distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/data_points_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ============== 保存预测的u和v数据 ==============
    # 创建网格坐标
    X_mesh, Y_mesh = np.meshgrid(x, y)

    # 展平为一维数组
    x_flat = X_mesh.flatten()
    y_flat = Y_mesh.flatten()
    u_flat = u_pred.flatten()
    v_flat = v_pred.flatten()

    # 组合为一个数组 [x, y, u, v]
    data_to_save = np.column_stack((x_flat, y_flat, u_flat, v_flat))

    # 保存为CSV格式（x,y,u,v）
    np.savetxt(f'{output_dir}/predicted_flow_data.csv', data_to_save,
               delimiter=',', header='x,y,u,v', comments='')

    print(f"预测数据已保存至目录: {output_dir}")
    print(f"合并格式数据: {output_dir}/predicted_flow_data.csv")
    print("Inverse problem solved and results saved!")