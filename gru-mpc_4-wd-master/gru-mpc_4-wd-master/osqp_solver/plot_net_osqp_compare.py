import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

from init_mpc import init_MPC
from init_mpc import mpc_config_to_numpy
from init_vehicle_sim import P_Vehicle
from init_steer_actuator_sim import init_steer_actuator_sim
from mpc2qp import formulate_qp_problem, zoh_discrete
from vehicle_model import vehicle_model
from osqp_solver import solve_mpc_osqp
from supervised_learning_net import MLP



def main():
    # 固定网络在 CPU 上推理
    device_net = 'cpu'

    # 车辆与 MPC 初始化
    vehicle = P_Vehicle(device='cuda' if torch.cuda.is_available() else 'cpu')
    init_steer_actuator_sim(vehicle)
    p_vehicle = vehicle.to_numpy()
    mpc = init_MPC(p_vehicle)
    mpc_config = mpc_config_to_numpy(mpc)

    Nx = mpc_config.Nx
    Nu = mpc_config.Nu
    Ts = mpc_config.Ts

    # 常量：曲率与纵向速度
    kappa = -0.01
    vx = 20
    N_STEPS = 200

    # 离散化系统（vx、车辆参数恒定，因此 Ad/Bd/Ed 固定）
    A, B, E = vehicle_model(p_vehicle, vx)
    Ad, Bd, Ed = zoh_discrete(A, B, E, Ts)
    Ed = Ed.flatten()  # (Nx,)

    # 初始状态（长度需为 Nx）
    x0 = np.ones(Nx) * 0.1
    x_net = x0.copy()
    x_osqp = x0.copy()

    # 结果记录
    x_net_hist = np.zeros((N_STEPS + 1, Nx))
    x_osqp_hist = np.zeros((N_STEPS + 1, Nx))
    u_net_hist = np.zeros((N_STEPS, Nu))
    u_osqp_hist = np.zeros((N_STEPS, Nu))
    x_net_hist[0] = x_net
    x_osqp_hist[0] = x_osqp

    # 加载训练好的网络（输入维度 = 2 + Nx，输出维度 = Nu）
    model = MLP(2 + Nx, Nu).to(device_net)
    try:
        state = torch.load('supervised_osqp_mlp.pth', map_location=device_net, weights_only=True)
    except TypeError:
        state = torch.load('supervised_osqp_mlp.pth', map_location=device_net)
    model.load_state_dict(state)
    model.eval()

    net_times = []
    osqp_times = []

    for t in range(N_STEPS):
        # ===== 网络控制（仅当前步 u） =====
        net_input = np.concatenate([[kappa, vx], x_net]).astype(np.float32)[None, :]
        net_input_t = torch.from_numpy(net_input).to(device_net)
        with torch.no_grad():
            t0 = time.perf_counter()
            u_net = model(net_input_t).cpu().numpy().reshape(-1)  # (Nu,)
            t1 = time.perf_counter()
        net_times.append(t1 - t0)
        u_net_hist[t] = u_net
        # x_{k+1} = Ad x_k + Bd u_k + Ed * kappa
        x_net = Ad @ x_net + Bd @ u_net + Ed * kappa
        x_net_hist[t + 1] = x_net

        # ===== OSQP-MPC（取第一步 u） =====
        t2 = time.perf_counter()
        H, q, Aeq, beq = formulate_qp_problem(mpc_config, p_vehicle, kappa, vx, x_osqp)
        osqp_loss, solution, status = solve_mpc_osqp(H, q, Aeq, beq)
        t3 = time.perf_counter()
        osqp_times.append(t3 - t2)

        if solution is not None:
            # 取第 0 步控制：其在解向量中的位置为紧跟第 0 步 x 之后的 Nu 个量
            u0 = solution[Nx:Nx + Nu]
        else:
            # 失败兜底：用 0 控制
            u0 = np.zeros(Nu)
        u_osqp_hist[t] = u0
        x_osqp = Ad @ x_osqp + Bd @ u0 + Ed * kappa
        x_osqp_hist[t + 1] = x_osqp

    # 误差度量
    mse_u = np.mean((u_net_hist - u_osqp_hist) ** 2)
    print(f'控制序列 MSE: {mse_u:.6e}')
    print(f'网络推理平均耗时: {np.mean(net_times) * 1000:.3f} ms/step')
    print(f'OSQP 平均耗时: {np.mean(osqp_times) * 1000:.3f} ms/step')

    # 作图对比：3x2 共6张图（vy, r, ey, ephi, delta, u）
    t_axis = np.arange(N_STEPS)
    t_axis_x = np.arange(N_STEPS + 1)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    state_names = ['vy', 'r', 'ey', 'ephi', 'delta']
    # 确保 Nx 至少包含前5个通道，若不足则用可用的通道并留空多余子图
    for i in range(5):
        ax = axes[i]
        if i < Nx:
            ax.plot(t_axis_x, x_net_hist[:, i], label=f'NN {state_names[i]}' if i < len(state_names) else f'NN x[{i}]', linewidth=2)
            ax.plot(t_axis_x, x_osqp_hist[:, i], label=f'OSQP {state_names[i]}' if i < len(state_names) else f'OSQP x[{i}]', linewidth=2, linestyle='--')
            title = state_names[i] if i < len(state_names) else f'x[{i}]'
            ax.set_title(f'{title}')
            ax.set_xlabel('step')
            ax.set_ylabel(title)
            ax.grid(True)
            ax.legend()
        else:
            ax.axis('off')

    # 控制量 u（假设至少有一个通道）
    ax_u = axes[5]
    if Nu >= 1:
        ax_u.plot(t_axis, u_net_hist[:, 0], label='NN u', linewidth=2)
        ax_u.plot(t_axis, u_osqp_hist[:, 0], label='OSQP u', linewidth=2, linestyle='--')
        ax_u.set_title('u')
        ax_u.set_xlabel('step')
        ax_u.set_ylabel('u')
        ax_u.grid(True)
        ax_u.legend()
    else:
        ax_u.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()


