from cal_KxKu import cal_KxKu
from vehicle_model import vehicle_model
import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
from scipy.sparse.linalg import spsolve
from mpc2qp import zoh_discrete


def solve_states_from_constraints(mpc_config, p_vehicle, kappa, vx, x0, u):

    # 计算连续系统并离散化 (矩阵在预测域内固定)
    A, B, E = vehicle_model(p_vehicle, vx)
    Ad, Bd, Ed = zoh_discrete(A, B, E, mpc_config.Ts)

    Nx = mpc_config.Nx
    Nu = mpc_config.Nu
    N  = mpc_config.N  # 期望的预测步数

    # 处理初始状态形状
    x0 = np.asarray(x0).reshape(-1)
    assert x0.shape[0] == Nx, f"x0 维度 {x0.shape[0]} 与 Nx={Nx} 不匹配"

    # 控制输入：假设外部已保证尺寸 (N,1)
    u = np.asarray(u).reshape(N, 1)

    # 预分配状态数组
    X = np.zeros((N + 1, Nx))
    X[0] = x0

    # 预计算扰动项
    w = Ed[:, 0] * kappa

    for k in range(N):
        uk = u[k].reshape(Nu, 1)  # (Nu,1)
        X[k+1] = (Ad @ X[k].reshape(Nx, 1) + Bd @ uk).reshape(-1) + w

    return X

if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import torch
    from init_mpc import init_MPC
    from init_mpc import mpc_config_to_numpy
    from init_vehicle_sim import P_Vehicle
    from init_steer_actuator_sim import init_steer_actuator_sim
    from osqp_solver import solve_mpc_osqp
    from osqp_solver import cal_osqp_mpc_loss
    import numpy as np
    import matplotlib.pyplot as plt

    vehicle = P_Vehicle(device='cuda' if torch.cuda.is_available() else 'cpu') 
    init_steer_actuator_sim(vehicle) 
    p_vehicle = vehicle.to_numpy() # 把车辆参数转化为numpy版本
    mpc = init_MPC(p_vehicle)
    mpc_config = mpc_config_to_numpy(mpc) # 把mpc初始化参数转化为numpy版本
    
    N = mpc_config.N #预测时域
    

    # 读取参数
    Vx_bar_batch = np.load('result/Vx_bar_batch.npy')
    kappa_batch = np.load('result/kappa_batch.npy')
    x_net_batch = np.load('result/x_net_batch.npy')
    u_net_batch = np.load('result/u_net_batch.npy')

    batch_size = Vx_bar_batch.shape[0]

    x_osqp_N_steps = []

    for i in range(batch_size):
        kappa = kappa_batch[i].item()
        Vx_bar = Vx_bar_batch[i,].item()
        x0 = x_net_batch[i,0,:]
        u_net = u_net_batch[i,:,:]
        x_osqp = solve_states_from_constraints(mpc_config, p_vehicle, kappa, Vx_bar, x0, u_net)
        x_osqp_N_steps.append(x_osqp)

    x_osqp_batch = np.asarray(x_osqp_N_steps)

    # x_osqp_batch和x_net_batch作差对比误差
    diff = x_osqp_batch - x_net_batch
    #print(np.mean(np.abs(diff)))

    # 将x_osqp_batch和u_net_batch放入solution，
    # 期望 solution 结构为 [x0, u0, x1, u1, ..., xN-1, uN-1, xN]
    Nx = mpc_config.Nx
    Nu = mpc_config.Nu
    N  = mpc_config.N

    num_vars = N * (Nx + Nu) + Nx
    solution_batch = np.zeros((batch_size, num_vars))
    for i in range(batch_size):
        base = 0
        for k in range(N):
            solution_batch[i, base:base + Nx] = x_osqp_batch[i, k, :]
            solution_batch[i, base + Nx:base + Nx + Nu] = u_net_batch[i, k, :]
            base += (Nx + Nu)
        # 末尾放入 x_N
        solution_batch[i, base:base + Nx] = x_osqp_batch[i, N, :]

    #print('solution_batch shape:', solution_batch.shape)

    # 计算损失（按 batch 使用对应的 kappa 与 Vx_bar）
    loss_batch = []
    for i in range(batch_size):
        solution = solution_batch[i, :]
        Vx_bar = Vx_bar_batch[i].item()
        kappa = kappa_batch[i].item()
        osqp_loss, mpc_loss = cal_osqp_mpc_loss(solution, mpc_config, p_vehicle, kappa, Vx_bar)
        loss_batch.append(mpc_loss)

    print(f"OSQP_mpc损失: {np.mean(loss_batch)}")

    # vehicle = P_Vehicle(device='cuda' if torch.cuda.is_available() else 'cpu') 
    # init_steer_actuator_sim(vehicle) 
    # p_vehicle = vehicle.to_numpy() # 把车辆参数转化为numpy版本
    # mpc = init_MPC(p_vehicle)
    # mpc_config = mpc_config_to_numpy(mpc) # 把mpc初始化参数转化为numpy版本
 
    # # 参数
    # vx = 10.0
    # kappa = 0.01

    # # 组合初始状态向量
    # x0 = np.concatenate([np.array([0.01, 0.01, 0.01, 0.01, 0.01])])

    # # 预测维度信息
    # N = mpc_config.N
    # Nx = mpc_config.Nx
    # Nu = mpc_config.Nu

    # u_net = np.load('result/u_net.npy')
    # #u_net = np.load('u_net_RK4.npy')
    # #u_net = 0.01*np.ones(N)

    # X = solve_states_from_constraints(mpc_config, p_vehicle, kappa, vx, x0, u_net)

    # np.save('result/x_osqp.npy', X)

    # steps = np.arange(N + 1)
    # rows = min(5, Nx)
    # fig, axes = plt.subplots(rows, 1, figsize=(7, 2.2 * rows), sharex=True)
    # if rows == 1:
    #     axes = [axes]
    # for i in range(rows):
    #     ax = axes[i]
    #     ax.plot(steps, X[:, i], label=f'x[{i}]')
    #     ax.set_ylabel(f'x[{i}]')
    #     ax.grid(True)
    #     ax.legend(loc='best')
    # fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    # plt.show()


    



