import scipy.sparse as sp
import osqp
import numpy as np
from cal_KxKu import cal_KxKu

def solve_mpc_osqp(H, q, Aeq, beq):
    
    P = sp.csc_matrix(H)
    A = sp.csc_matrix(Aeq)
    l = beq
    u = beq
    
    prob = osqp.OSQP()
    prob.setup(P, q, A, l, u, verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=10000)
    
    # 求解问题
    sol = prob.solve()
    loss = sol.info.obj_val
    sol_time = sol.info.solve_time

    status = sol.info.status
    return loss, sol.x,status

def cal_osqp_mpc_loss(solution, mpc_config, p_vehicle, kappa, vx):

    # 获取MPC参数
    Q = mpc_config.Q
    R = mpc_config.R
    QN = mpc_config.QN
    Q_diag = np.diag(Q.flatten())
    R_diag = np.diag(R.flatten())
    QN_diag = np.diag(QN.flatten())
    N = mpc_config.N
    Nx = mpc_config.Nx
    Nu = mpc_config.Nu
    decay = mpc_config.decay_factor
    
    # 计算参考轨迹
    Kx, Ku = cal_KxKu(vx, p_vehicle)
    x_ref = Kx * kappa
    u_ref = Ku * kappa
    
    # 已经是numpy数组
    x_ref = x_ref.flatten()
    u_ref = u_ref.flatten()
    
    # 从solution中提取状态和控制轨迹
    x_traj = np.zeros((N+1, Nx))
    u_traj = np.zeros((N, Nu))
    
    # 提取轨迹：直接从solution中提取所有状态和控制
    for k in range(N):
        x_idx = k * (Nx + Nu)
        u_idx = x_idx + Nx
        x_traj[k, :] = solution[x_idx:x_idx+Nx]  # 当前时刻状态
        u_traj[k, :] = solution[u_idx:u_idx+Nu]  # 当前时刻控制
    
    # 提取最终状态 (第N+1个状态在solution的最后)
    x_traj[N, :] = solution[N * (Nx + Nu):N * (Nx + Nu) + Nx]
    
    # 计算代价函数
    quadratic_cost = 0
    linear_cost = 0
    constant_cost = 0
    
    for k in range(N):
        wk = decay ** k
        # 当前状态和控制
        xk = x_traj[k, :]
        uk = u_traj[k, :]
        
        # 二次项: 1/2 * (x^T Q x + u^T R u)
        quadratic_cost += 0.5 * wk * (xk.T @ Q_diag @ xk + uk.T @ R_diag @ uk)
        
        # 线性项: -(x^T Q x_ref + u^T R u_ref)
        linear_cost -= wk * (xk.T @ Q_diag @ x_ref + uk.T @ R_diag @ u_ref)
        
        # 常数项: x_ref^T Q x_ref + u_ref^T R u_ref
        constant_cost += wk * (x_ref.T @ Q_diag @ x_ref + u_ref.T @ R_diag @ u_ref)
    
    # 终端代价 (k=N)
    xN = x_traj[N, :]
    quadratic_cost += 0.5 * (xN.T @ QN_diag @ xN)
    linear_cost -= xN.T @ QN_diag @ x_ref
    constant_cost += x_ref.T @ QN_diag @ x_ref
    
    # OSQP目标函数值 = 1/2 * x^T H x + q^T x + const
    # 其中 H x + q 对应我们的二次项和线性项
    osqp_loss = quadratic_cost + linear_cost
    
    # MPC原始目标函数值 = osqp_loss (因为OSQP求解的就是原始MPC问题)
    mpc_loss = osqp_loss * 2 + constant_cost
    
    return osqp_loss, mpc_loss

if __name__ == '__main__': 
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import torch
    from init_mpc import init_MPC
    from init_mpc import mpc_config_to_numpy
    from init_vehicle_sim import P_Vehicle
    from init_steer_actuator_sim import init_steer_actuator_sim
    from mpc2qp import formulate_qp_problem
    from mpc2qp import zoh_discrete
    import numpy as np
    import matplotlib.pyplot as plt

    vehicle = P_Vehicle(device='cuda' if torch.cuda.is_available() else 'cpu') 
    init_steer_actuator_sim(vehicle) 
    p_vehicle = vehicle.to_numpy() # 把车辆参数转化为numpy版本
    mpc = init_MPC(p_vehicle)
    mpc_config = mpc_config_to_numpy(mpc) # 把mpc初始化参数转化为numpy版本
    # 数据集
    vx = 10 # 车速
    kappa = 0.01 # 曲率
    x0 = np.concatenate([np.array([0.01, 0.01, 0.01, 0.01, 0.01])]) # 状态量
    # 构建QP问题
    H, q, Aeq, beq = formulate_qp_problem(mpc_config, p_vehicle, kappa, vx, x0)
    # osqp求解器返回的损失和解
    osqp_loss, solution, status = solve_mpc_osqp(H, q, Aeq, beq)
    print("求解状态:",status)
    # 手动计算osqp的损失和mpc问题的损失
    osqp_loss_manual, mpc_loss_manual = cal_osqp_mpc_loss(solution, mpc_config, p_vehicle, kappa, vx)

    print(f"手动计算osqp损失: {osqp_loss_manual:.6f}")
    print(f"手动计算osqp_MPC损失: {mpc_loss_manual:.6f}")
    print(f"osqp损失: {osqp_loss:.6f}")
    
    # 保存osqp的solution到工作路径
    np.save('solution_osqp.npy', solution)
    print(f"已保存 solution_osqp 到: solution_osqp.npy")
    print(solution.shape)

    # 绘制状态量和控制量轨迹图
    # 计算参考值
    Kx, Ku = cal_KxKu(vx, p_vehicle)
    x_ref = Kx * kappa
    u_ref = Ku * kappa
    # 从solution中提取轨迹
    N = mpc_config.N
    Nx = mpc_config.Nx 
    Nu = mpc_config.Nu
    
    x_traj = np.zeros((N+1, Nx))
    u_traj = np.zeros((N, Nu))
    
    # 解析OSQP解
    for k in range(N):
        x_idx = k * (Nx + Nu)
        u_idx = x_idx + Nx
        x_traj[k, :] = solution[x_idx:x_idx+Nx]
        u_traj[k, :] = solution[u_idx:u_idx+Nu]
    # 终端状态
    x_traj[N, :] = solution[N * (Nx + Nu):N * (Nx + Nu) + Nx]
    #保存u_traj为.npy
    np.save('u_traj.npy', u_traj)
    

    # 创建时间轴
    time_steps_x = np.arange(N+1) * mpc_config.Ts
    time_steps_u = np.arange(N) * mpc_config.Ts
    
    # 状态量和控制量名称
    state_names = ['vy (m/s)', 'r (rad/s)', 'ey (m)', 'ephi (rad)', 'delta (rad)']
    control_names = ['u (rad)']
    
    # 创建子图: 5个状态 + 1个控制 = 6个子图
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # 绘制5个状态量
    for i in range(5):
        ax = axes[i]
        
        # 参考值 (水平线)
        ref_value = x_ref[i]
        ax.axhline(y=ref_value, color='green', linestyle='--', linewidth=2, 
                  label=f'Reference: {ref_value:.4f}')
        
        # OSQP轨迹
        ax.plot(time_steps_x, x_traj[:, i], 'b-o', 
               linewidth=2, markersize=4)
        
        ax.set_title(f'State {i+1}: {state_names[i]}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(state_names[i])
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 设置y轴范围，使参考线更明显
        y_data = x_traj[:, i]
        y_range = max(abs(y_data.max()), abs(y_data.min()), abs(ref_value))
        if y_range > 0:
            ax.set_ylim([-y_range*1.2, y_range*1.2])
    
    # 绘制控制量
    ax = axes[5]
    
    # 参考控制量 (水平线)
    u_ref_value = u_ref[0]
    ax.axhline(y=u_ref_value, color='green', linestyle='--', linewidth=2, 
              label=f'Reference: {u_ref_value:.4f}')
    
    # OSQP控制轨迹
    ax.plot(time_steps_u, u_traj[:, 0], 'b-o', 
           linewidth=2, markersize=4, label='OSQP Solution')
    
    ax.set_title(f'Control: {control_names[0]}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(control_names[0])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 设置y轴范围
    u_data = u_traj[:, 0]
    u_range = max(abs(u_data.max()), abs(u_data.min()), abs(u_ref_value))
    if u_range > 0:
        ax.set_ylim([-u_range*1.2, u_range*1.2])
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('osqp_mpc_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n图像已保存为 'osqp_mpc_results.png'")
    print("绘图完成！")
