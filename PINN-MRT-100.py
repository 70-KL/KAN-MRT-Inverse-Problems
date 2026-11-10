import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
from datetime import datetime
import json

# --------------------------1.设置设备和随机种子
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# -------------------------------------------------------------------------------------------

# --------------------------2.神经网络定义
class NN_eq(nn.Module):
    def __init__(self, input_dim=2, output_dim=3, hidden_dim=60, num_hidden=3, activation='tanh', initial_nu_hat=0.01):

        super(NN_eq, self).__init__()
        print("  - 逆问题 MRT 网络架构定义:")
        print(f"    输入层维度: {input_dim}")
        print(f"    隐藏层数量: {num_hidden}, 每层神经元数: {hidden_dim}")
        print(f"    输出层维度: {output_dim}")
        print(f"    激活函数: {activation}")

        self.nu_hat = nn.Parameter(torch.tensor([initial_nu_hat], requires_grad=True).float().to(device))
        print(f"    可学习参数: 黏度 nu_hat (初始值: {initial_nu_hat:.4f})")

        # --- 网络架构定义 (与正问题相同) ---
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_hidden - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sin':
            self.activation = torch.sin
        else:
            raise ValueError(f"不支持的激活函数 '{activation}'。请选择 'tanh' 或 'sin'。")

        self.history = {
            'total_loss': [], 'loss_pde': [], 'loss_bc': [], 'loss_data': [], 'nu_hat': []  # 新增项
        }
        print("  - 逆问题模型初始化完成。")

    def forward(self, x):  # 修改：只接受一个输入张量 x
        x_coord = x[:, 0:1]  # 提取x坐标
        y_coord = x[:, 1:2]  # 提取y坐标

        input_combined = torch.cat([x_coord, y_coord], dim=-1)  # 合并为一个张量
        out = input_combined
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        out = self.layers[-1](out)
        return out

    def get_tau(self):
        return 3.0 * self.nu_hat + 0.5  # 修正：使用self.nu_hat

    def get_relaxation_matrix(self):
        tau = self.get_tau()

        tau_rho = 1.0  # 密度松弛时间
        tau_e = 1.0 / 0.8  # 能量松弛时间
        tau_zita = 1.0 / 0.8  # 能量通量松弛时间
        tau_j = 1.0  # 动量松弛时间
        tau_q = 1.0 / 1.1  # 动量通量松弛时间

        # 构建松弛矩阵 (确保梯度能正确流向nu_hat)
        S_vals = torch.cat([
            1.0 / tau_rho * torch.ones(1, device=device),
            1.0 / tau_e * torch.ones(1, device=device),
            1.0 / tau_zita * torch.ones(1, device=device),
            1.0 / tau_j * torch.ones(2, device=device),
            1.0 / tau_q * torch.ones(2, device=device),
            1.0 / tau * torch.ones(2, device=device)
        ])

        S = torch.diag(S_vals)
        MSM = torch.matmul(M_inv, torch.matmul(S, M))
        return MSM


class NN_neq(nn.Module):
    def __init__(self, input_dim=2, output_dim=9, hidden_dim=60, num_hidden=3, activation='tanh'):

        super(NN_neq, self).__init__()
        print("  - 网络架构定义:")
        print(f"    输入层维度: {input_dim}")
        print(f"    隐藏层数量: {num_hidden}, 每层神经元数: {hidden_dim}")
        print(f"    输出层维度: {output_dim}")
        print(f"    激活函数: {activation}")

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_hidden - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        # 选择激活函数
        if activation == 'tanh':
            self.activation = torch.tanh  # 使用双曲正切函数 tanh
        elif activation == 'sin':
            self.activation = torch.sin  # 使用正弦函数 sin (有时用于需要周期性的问题)
        else:
            raise ValueError(f"不支持的激活函数 '{activation}'。请选择 'tanh' 或 'sin'。")
        print("  - 模型初始化完成。")

    def forward(self, x):  # 修改：只接受一个输入张量 x
        x_coord = x[:, 0:1]  # 提取x坐标
        y_coord = x[:, 1:2]  # 提取y坐标

        input_combined = torch.cat([x_coord, y_coord], dim=-1)  # 合并为一个张量
        out = input_combined
        for layer in self.layers[:-1]:
            out = layer(out)
            out = self.activation(out)
        out = self.layers[-1](out)
        return out

# -------------------------------------------------------------------------------------------

# --------------------------3.加载真实数据集
def load_observation_data(file_path, num_points=None):
    """加载观测数据并采样"""
    data = pd.read_csv(file_path)
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


# -------------------------------------------------------------------------------------------

# --------------------------4.采样函数
def sample(N, coords, velocity_func):
    x0, y0 = coords[0]
    x1, y1 = coords[1]
    if x0 == x1:  # 垂直边界
        x = np.full((N, 1), x0)
        y = np.random.uniform(y0, y1, (N, 1))
    elif y0 == y1:  # 水平边界
        x = np.random.uniform(x0, x1, (N, 1))
        y = np.full((N, 1), y0)
    else:
        raise ValueError("Coordinates must define a horizontal or vertical line")
    X = torch.tensor(np.hstack((x, y)), requires_grad=True, dtype=torch.float32).to(device)
    u_v = velocity_func(None)
    return X, torch.tensor(u_v[0], dtype=torch.float32).to(device), torch.tensor(u_v[1], dtype=torch.float32).to(device)


# -------------------------------------------------------------------------------------------

# --------------------------5.PDE损失，DATA损失，边界损失
# 计算平衡分布函数
def compute_f_eq(rho, u, v):
    u = u.squeeze(-1)
    v = v.squeeze(-1)
    U = torch.stack([u, v], dim=-1)
    e_U = torch.einsum('ij,kj->ki', e, U)
    U_sq = torch.sum(U ** 2, dim=1)
    f_eq = w * rho.squeeze()[:, None] * (1 + 3 * e_U + 4.5 * e_U ** 2 - 1.5 * U_sq[:, None])
    return f_eq


# 修改物理损失函数
def physics_loss_LBM(X, model_eq, model_neq, training_mode=True):
    rho_U = model_eq(X)
    rho, u, v = rho_U[:, 0:1], rho_U[:, 1:2], rho_U[:, 2:3]
    f_eq = compute_f_eq(rho, u, v)
    f_neq = model_neq(X)
    f = f_eq + f_neq

    # 获取基于预测nu_hat的松弛矩阵
    MSM = model_eq.get_relaxation_matrix()

    xi = torch.zeros_like(f)
    residual = torch.zeros_like(f)
    for i in range(9):
        grad_f_i = torch.autograd.grad(f[:, i], X,
                                       grad_outputs=torch.ones_like(f[:, i]),
                                       create_graph=True)[0]
        xi_dot_grad_f_i = e[i, 0] * grad_f_i[:, 0] + e[i, 1] * grad_f_i[:, 1]
        xi[:, i] = xi_dot_grad_f_i
        residual[:, i] = xi_dot_grad_f_i + torch.matmul(f - f_eq, MSM)[:, i]

    return torch.sqrt(torch.mean(residual ** 2) + 1e-10)


# 数据损失函数
def data_loss(model_eq, x_data, y_data, u_data, v_data, training_mode=True):
    """计算模型预测与观测数据的损失"""
    X_data = torch.cat([x_data, y_data], dim=1)
    rho_U_pred = model_eq(X_data)
    u_pred, v_pred = rho_U_pred[:, 1:2], rho_U_pred[:, 2:3]
    loss_u = torch.mean((u_pred - u_data) ** 2)
    loss_v = torch.mean((v_pred - v_data) ** 2)
    loss = loss_u + loss_v

    return loss


# 边界损失函数 - 修复了L_fbc计算逻辑
def boundary_loss_per_boundary(X_boundary, u_b, v_b, n, model_eq, model_neq, e, w, M, M_inv, device,
                               training_mode=True):
    n = n.to(dtype=torch.float32)
    rho_U = model_eq(X_boundary)
    rho, u, v = rho_U[:, 0:1], rho_U[:, 1:2], rho_U[:, 2:3]

    # 质量和动量守恒边界条件损失
    L_mbc = torch.mean((rho - 1) ** 2) + torch.mean((u - u_b) ** 2) + torch.mean((v - v_b) ** 2)

    f_eq = compute_f_eq(rho, u, v)
    tau = 0.53
    tau_rho = 1.0  # 密度松弛时间 (通常设为1)
    tau_e = 1.0 / 0.8  # 能量松弛时间
    tau_zita = 1.0 / 0.8  # 能量通量松弛时间
    tau_j = 1.0  # 动量松弛时间
    tau_q = 1.0 / 1.1  # 动量通量松弛时间

    # 构建松弛矩阵
    S = torch.diag(torch.tensor([
        1 / tau_rho, 1 / tau_e, 1 / tau_zita,
        1 / tau_j, 1 / tau_q,
        1 / tau_j, 1 / tau_q,
        1 / tau, 1 / tau
    ], dtype=torch.float32).to(device))

    MSM = torch.matmul(M_inv, torch.matmul(S, M))

    f_neq = model_neq(X_boundary)
    f = f_eq + f_neq

    e_dot_n = e @ n
    incoming = [i for i in range(9) if e_dot_n[i] > 0]
    outgoing = [i for i in range(9) if e_dot_n[i] <= 0]

    # 非平衡分布函数边界条件损失
    L_fbc = 0.0
    xi_dot_feq = torch.zeros_like(f)

    # 修复：计算incoming方向的f_neq
    for i in incoming:
        grad_f_eq_i = torch.autograd.grad(f_eq[:, i], X_boundary,
                                          grad_outputs=torch.ones_like(f_eq[:, i]),
                                          create_graph=True)[0]
        xi_dot_grad_f_eq_i = e[i, 0] * grad_f_eq_i[:, 0] + e[i, 1] * grad_f_eq_i[:, 1]
        xi_dot_feq[:, i] = xi_dot_grad_f_eq_i

        # 计算非平衡分布函数的边界条件
        f_neq_star = -torch.matmul(xi_dot_feq, MSM)
        L_fbc += torch.mean((f_neq[:, i] - f_neq_star[:, i]) ** 2)

    # 流场方程边界条件损失
    L_Ebc = 0.0
    xi_dot_f = torch.zeros_like(f)

    # 使用不同的索引变量j避免混淆
    for j in outgoing:
        grad_f_j = torch.autograd.grad(f[:, j], X_boundary,
                                       grad_outputs=torch.ones_like(f[:, j]),
                                       create_graph=True)[0]
        xi_dot_grad_f_j = e[j, 0] * grad_f_j[:, 0] + e[j, 1] * grad_f_j[:, 1]
        xi_dot_f[:, j] = xi_dot_grad_f_j

        # 计算残差
        R_j = xi_dot_f[:, j] + torch.matmul(f - f_eq, MSM)[:, j]
        L_Ebc += torch.mean(R_j ** 2)

    L_bc = L_mbc + L_fbc + L_Ebc

    return L_bc


def boundary_loss_LBM(model_eq, model_neq, e, w, M, M_inv, device, training_mode=True):
    L_bc_total = 0.0
    for X_b, u_b, v_b, n in [(X_top, u_top, v_top, torch.tensor([0, 1], device=device)),
                             (X_left, u_left, v_left, torch.tensor([-1, 0], device=device)),
                             (X_right, u_right, v_right, torch.tensor([1, 0], device=device)),
                             (X_bottom, u_bottom, v_bottom, torch.tensor([0, -1], device=device))]:
        L_bc = boundary_loss_per_boundary(X_b, u_b, v_b, n, model_eq, model_neq, e, w, M, M_inv, device, training_mode)
        L_bc_total += L_bc

    return L_bc_total


# -------------------------------------------------------------------------------------------

# ---------------------------6.可视化宏观量数据和绝对误差，train-loss
def visualize_results(model_eq, results_dir, epoch=None):
    os.makedirs(results_dir, exist_ok=True)
    N = 100
    x, y = np.linspace(0, 1, N), np.linspace(0, 1, N)
    X_grid, Y_grid = np.meshgrid(x, y)
    XY = torch.tensor(np.hstack((X_grid.flatten()[:, None], Y_grid.flatten()[:, None])), requires_grad=True).float().to(
        device)

    rho_U_pred = model_eq(XY)
    rho, u, v = rho_U_pred[:, 0:1], rho_U_pred[:, 1:2], rho_U_pred[:, 2:3]
    u_pred = u.detach().cpu().numpy().reshape(N, N)
    v_pred = v.detach().cpu().numpy().reshape(N, N)
    rho_pred = rho.detach().cpu().numpy().reshape(N, N)
    p_pred = rho_pred / 3  # 假设 p = rho / 3

    psi_pred = np.zeros((N, N))
    for i in range(1, N):
        psi_pred[i, :] = psi_pred[i - 1, :] + v_pred[i, :] * (y[1] - y[2])
    for j in range(1, N):
        psi_pred[:, j] = psi_pred[:, j - 1] - u_pred[:, j] * (x[1] - x[2])

    # 绘制结果
    fields = {'u velocity': u_pred, 'v velocity': v_pred, 'Pressure (p)': p_pred, 'Stream Function (psi)': psi_pred}
    for title, field in fields.items():
        plt.figure(figsize=(6, 5))
        plt.contourf(X_grid, Y_grid, field, levels=50, cmap='jet')
        plt.colorbar(label=title)
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        suffix = f"_epoch_{epoch}" if epoch is not None else ""
        plt.savefig(os.path.join(results_dir, f"{title.replace(' ', '_')}{suffix}.png"), dpi=300)
        plt.close()

    # 可视化损失
    plt.figure(figsize=(12, 6))
    plt.plot(model_eq.history['total_loss'], label='Total Loss')
    plt.plot(model_eq.history['loss_pde'], label='PDE Loss')  # 修正：使用正确的键名
    plt.plot(model_eq.history['loss_data'], label='Data Loss')  # 修正：使用正确的键名
    plt.plot(model_eq.history['loss_bc'], label='BC Loss')  # 修正：使用正确的键名
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss History')
    plt.grid(True)
    suffix = f"_epoch_{epoch}" if epoch is not None else ""
    plt.savefig(os.path.join(results_dir, f"loss_history{suffix}.png"), dpi=300)
    plt.close()

    # 可视化黏度预测过程
    plt.figure(figsize=(12, 6))
    plt.plot(model_eq.history['nu_hat'])  # 修正：使用正确的键名
    plt.axhline(y=0.01, color='r', linestyle='--', label='True nu')
    plt.xlabel('Epoch')
    plt.ylabel('Predicted nu')
    plt.title('Viscosity Prediction History')
    plt.legend()
    plt.grid(True)
    suffix = f"_epoch_{epoch}" if epoch is not None else ""
    plt.savefig(os.path.join(results_dir, f"nu_history{suffix}.png"), dpi=300)
    plt.close()
# -------------------------------------------------------------------------------------------

# ---------------------------------------7.保存nu，train-loss，可视化数据u，v
# 创建结果保存目录
def create_results_directory():
    """创建结果保存目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/inverse_MRT_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


# 保存训练历史
def save_training_history(model_eq, results_dir, params, epoch=None):
    os.makedirs(results_dir, exist_ok=True)
    history = model_eq.history
    for key in history.keys():
        if isinstance(history[key], list) and all(isinstance(x, (int, float)) for x in history[key]):
            history[key] = [float(x) for x in history[key]]
        elif isinstance(history[key], (int, float)):
            history[key] = float(history[key])

    history['final_nu'] = float(model_eq.nu_hat.item())
    history['true_nu'] = 0.01

    suffix = f"_epoch_{epoch}" if epoch is not None else ""
    history_path = os.path.join(results_dir, f'training_history{suffix}.json')
    params_path = os.path.join(results_dir, f'parameters{suffix}.json')

    # 保存历史
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)

    # 保存参数（每次保存时都写入当前参数）
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)

    print(f"训练历史已保存到: {history_path}")
    print(f"训练参数已保存至: {params_path}")

# 保存预测数据
def save_prediction_data(model_eq, results_dir, N=100, epoch=None):
    os.makedirs(results_dir, exist_ok=True)
    x, y = np.linspace(0, 1, N), np.linspace(0, 1, N)
    X_grid, Y_grid = np.meshgrid(x, y)
    XY = torch.tensor(np.hstack((X_grid.flatten()[:, None], Y_grid.flatten()[:, None])),
                      requires_grad=True).float().to(device)
    model_eq.eval()
    with torch.no_grad():
         rho_U_pred = model_eq(XY)
    rho, u, v = rho_U_pred[:, 0:1], rho_U_pred[:, 1:2], rho_U_pred[:, 2:3]
    u_pred = u.detach().cpu().numpy().reshape(N, N)
    v_pred = v.detach().cpu().numpy().reshape(N, N)
    speed = np.sqrt(u_pred ** 2 + v_pred ** 2)
    rho_pred = rho.detach().cpu().numpy().reshape(N, N)
    p_pred = rho_pred / 3  # 假设 p = rho / 3

    # 创建坐标网格
    x_coords = np.tile(x, N)
    y_coords = np.repeat(y, N)

    # 提取中心线数据
    x_center_idx = np.abs(x - 0.5).argmin()
    y_center_idx = np.abs(y - 0.5).argmin()

    u_centerline_y = u_pred[:, x_center_idx]  # x=0.5处的u速度剖面
    v_centerline_x = v_pred[y_center_idx, :]  # y=0.5处的v速度剖面

    # 创建完整网格数据集
    grid_data = {
        'X': x_coords,
        'Y': y_coords,
        'u': u_pred.flatten(),
        'v': v_pred.flatten(),
        'speed': speed.flatten(),
        'rho': rho_pred.flatten(),
        'p': p_pred.flatten()
    }

    # 创建中心线数据集
    centerline_data = {
        'y_coord': y,  # y坐标
        'u_at_x=0.5': u_centerline_y,  # x=0.5处的u速度随y变化
        'x_coord': x,  # x坐标
        'v_at_y=0.5': v_centerline_x  # y=0.5处的v速度随x变化
    }

    # 保存到CSV（分为两个文件）
    suffix = f"_epoch_{epoch}" if epoch is not None else ""

    # 保存完整网格数据
    grid_file = os.path.join(results_dir, f"grid_data{suffix}.csv")
    pd.DataFrame(grid_data).to_csv(grid_file, index=False)

    # 保存中心线数据
    centerline_file = os.path.join(results_dir, f"centerline_data{suffix}.csv")
    pd.DataFrame(centerline_data).to_csv(centerline_file, index=False)

    print(f"网格数据已保存到: {grid_file}")
    print(f"中心线数据已保存到: {centerline_file}")

    return grid_file, centerline_file
#--------------------------------------------------------------------------

# ==============================================================================
# 主执行流程
# ==============================================================================
if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("逆问题主执行流程开始")
    print("=" * 60)

    # --------1.加载观测数据
    x_data, y_data, u_data, v_data = load_observation_data("num-100.csv", num_points=1000)
    x_data_tensor = torch.tensor(x_data, dtype=torch.float32, requires_grad=True).to(device)
    y_data_tensor = torch.tensor(y_data, dtype=torch.float32, requires_grad=True).to(device)
    u_data_tensor = torch.tensor(u_data, dtype=torch.float32).to(device)
    v_data_tensor = torch.tensor(v_data, dtype=torch.float32).to(device)

    # --------2.随机种子
    seed_torch(42)

    # --------3.D2Q9 模型参数和神经网络模型设置
    e = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                      [1, 1], [-1, 1], [-1, -1], [1, -1]],
                     dtype=torch.float32).to(device)
    w = torch.tensor([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
                     dtype=torch.float32).to(device)

    M = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [-4, -1, -1, -1, -1, 2, 2, 2, 2],
        [4, -2, -2, -2, -2, 1, 1, 1, 1],
        [0, 1, 0, -1, 0, 1, -1, -1, 1],
        [0, -2, 0, 2, 0, 1, -1, -1, 1],
        [0, 0, 1, 0, -1, 1, 1, -1, -1],
        [0, 0, -2, 0, 2, 1, 1, -1, -1],
        [0, 1, -1, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 1, -1]
    ], dtype=torch.float32).to(device)

    M_inv = torch.tensor([
        [1 / 9.0, -1 / 9.0, 1 / 9.0, 0, 0, 0, 0, 0, 0],
        [1 / 9.0, -1 / 36.0, -1 / 18.0, 1 / 6.0, -1 / 6.0, 0, 0, 1 / 4.0, 0],
        [1 / 9.0, -1 / 36.0, -1 / 18.0, 0, 0, 1 / 6.0, -1 / 6.0, -1 / 4.0, 0],
        [1 / 9.0, -1 / 36.0, -1 / 18.0, -1 / 6.0, 1 / 6.0, 0, 0, 1 / 4.0, 0],
        [1 / 9.0, -1 / 36.0, -1 / 18.0, 0, 0, -1 / 6.0, 1 / 6.0, -1 / 4.0, 0],
        [1 / 9.0, 1 / 18.0, 1 / 36.0, 1 / 6.0, 1 / 12.0, 1 / 6.0, 1 / 12.0, 0, 1 / 4.0],
        [1 / 9.0, 1 / 18.0, 1 / 36.0, -1 / 6.0, -1 / 12.0, 1 / 6.0, 1 / 12.0, 0, -1 / 4.0],
        [1 / 9.0, 1 / 18.0, 1 / 36.0, -1 / 6.0, -1 / 12.0, -1 / 6.0, -1 / 12.0, 0, 1 / 4.0],
        [1 / 9.0, 1 / 18.0, 1 / 36.0, 1 / 6.0, 1 / 12.0, -1 / 6.0, -1 / 12.0, 0, -1 / 4.0]
    ], dtype=torch.float32).to(device)

    INITIAL_nu = np.random.rand() * 0.1  # 生成[0, 0.1]范围内的随机初始nu值
    print(f"开始训练逆问题模型，初始 nu = {INITIAL_nu:.6f}")
    model_eq = NN_eq().to(device)
    model_neq = NN_neq().to(device)

    # --------4.数据采样
    N_f = 20000
    X_f = torch.tensor(np.hstack((np.random.uniform(0, 1, (N_f, 1)), np.random.uniform(0, 1, (N_f, 1)))),
                       requires_grad=True).float().to(device)
    N_b = 5000
    bc1_coords = np.array([[0.0, 1.0], [1.0, 1.0]])
    bc2_coords = np.array([[0.0, 0.0], [0.0, 1.0]])
    bc3_coords = np.array([[1.0, 0.0], [1.0, 1.0]])
    bc4_coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    X_top, u_top, v_top = sample(N_b, bc1_coords, lambda x: np.array([1.0, 0.0]))
    X_left, u_left, v_left = sample(N_b, bc2_coords, lambda x: np.array([0.0, 0.0]))
    X_right, u_right, v_right = sample(N_b, bc3_coords, lambda x: np.array([0.0, 0.0]))
    X_bottom, u_bottom, v_bottom = sample(N_b, bc4_coords, lambda x: np.array([0.0, 0.0]))

    # --------5.优化器设置和优化器参数
    # 修正Adam优化器的参数设置
    optimizer_adam = optim.Adam(
        [
            {'params': [p for n, p in model_eq.named_parameters()], 'lr': 1e-3},  # 修正：只传递参数p
            {'params': model_neq.parameters(), 'lr': 1e-3}  # 保持不变
        ],
        lr=1e-3  # 默认学习率，会被上面的特定设置覆盖
    )

    # 修正LBFGS优化器的参数设置
    optimizer_lbfgs = optim.LBFGS(
        [p for n, p in model_eq.named_parameters()] + list(model_neq.parameters()),  # 修正：只传递参数p
        lr=1e-2, max_iter=100,
        max_eval=200,
        tolerance_grad=1e-11, tolerance_change=1e-11, history_size=100,
        line_search_fn='strong_wolfe'
    )

    # --------6.创建结果目录
    results_dir = create_results_directory()

    # --------7.其他训练模型参数
    epochs = 80000  # 增加训练轮次
    w_pde = 1.0
    w_data = 1.0  # 数据损失权重，可调整
    w_bc = 1.0
    print_freq = 1000
    save_data_freq = 20000
    print(f"保存结果到: {results_dir}")

    # 定义训练参数
    params = {
        'epochs': epochs,
        'w_pde': w_pde,
        'w_data': w_data,
        'w_bc': w_bc,
        'INITIAL_nu': INITIAL_nu,
        'N_f': N_f,
        'N_b': N_b,
        'lr_adam': 1e-3,
        'lr_lbfgs': 1e-2
    }

    start_time = time.time()
    for epoch in range(epochs):
        model_eq.train()  # 确保模型处于训练模式
        model_neq.train()
        optimizer_adam.zero_grad()

        loss_pde = physics_loss_LBM(X_f, model_eq, model_neq)
        loss_data = data_loss(model_eq, x_data_tensor, y_data_tensor, u_data_tensor, v_data_tensor)
        loss_bc = boundary_loss_LBM(model_eq, model_neq, e, w, M, M_inv, device)  # 修正：传入参数
        total_loss = w_pde * loss_pde + w_data * loss_data + w_bc * loss_bc

        total_loss.backward()

        # 修正：分别裁剪两个模型的梯度
        torch.nn.utils.clip_grad_norm_(model_eq.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model_neq.parameters(), max_norm=1.0)

        optimizer_adam.step()

        # 记录nu值和梯度
        current_nu = model_eq.nu_hat.item()  # 修正：使用nu_hat

        # 统一记录所有损失
        model_eq.history['total_loss'].append(total_loss.item())
        model_eq.history['loss_pde'].append(loss_pde.item())  # 修正：使用正确的键名
        model_eq.history['loss_data'].append(loss_data.item())  # 修正：使用正确的键名
        model_eq.history['loss_bc'].append(loss_bc.item())  # 修正：使用正确的键名
        model_eq.history['nu_hat'].append(current_nu)  # 修正：使用正确的键名

        if epoch % print_freq == 0:
            print(
                f'Epoch: {epoch}, Total Loss: {total_loss.item():.6f}, PDE Loss: {loss_pde.item():.6f}, Data Loss: {loss_data.item():.6f}, BC Loss: {loss_bc.item():.6f}, nu: {current_nu:.8f}')

        if epoch % save_data_freq == 0 and epoch > 0:
            save_prediction_data(model_eq, results_dir, epoch=epoch)
            save_training_history(model_eq, results_dir, params, epoch=epoch)
            visualize_results(model_eq, results_dir, epoch=epoch)


    # LBFGS优化
    def closure():
        model_eq.train()  # 确保模型处于训练模式
        model_neq.train()
        optimizer_lbfgs.zero_grad()

        loss_pde = physics_loss_LBM(X_f, model_eq, model_neq, training_mode=False)
        loss_data = data_loss(model_eq, x_data_tensor, y_data_tensor, u_data_tensor, v_data_tensor,
                              training_mode=False)
        loss_bc = boundary_loss_LBM(model_eq, model_neq, e, w, M, M_inv, device, training_mode=False)  # 修正：传入参数
        total_loss = w_pde * loss_pde + w_data * loss_data + w_bc * loss_bc
        total_loss.backward()

        # 记录LBFGS优化过程中的损失
        current_nu = model_eq.nu_hat.item()  # 修正：使用nu_hat

        model_eq.history['total_loss'].append(total_loss.item())
        model_eq.history['loss_pde'].append(loss_pde.item())  # 修正：使用正确的键名
        model_eq.history['loss_data'].append(loss_data.item())  # 修正：使用正确的键名
        model_eq.history['loss_bc'].append(loss_bc.item())  # 修正：使用正确的键名
        model_eq.history['nu_hat'].append(current_nu)  # 修正：使用正确的键名

        return total_loss


    print("开始LBFGS优化...")
    start_time = time.time()
    optimizer_lbfgs.step(closure)
    print(f'Training time with LBFGS: {time.time() - start_time:.2f} seconds')

    save_prediction_data(model_eq, results_dir, epoch=None)
    save_training_history(model_eq, results_dir, params, epoch=None)
    visualize_results(model_eq, results_dir, epoch=None)

    # 输出最终黏度
    predicted_nu = model_eq.nu_hat.item()  # 修正：使用nu_hat
    print(f"Predicted viscosity nu = {predicted_nu:.8f}")
    print(f"True viscosity = 0.01")
    print(f"Relative error = {abs(predicted_nu - 0.01) / 0.01 * 100:.2f}%")

    print(f"\n逆问题训练完成，所有结果保存在: {results_dir}")