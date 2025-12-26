import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from cal_KxKu_batch import batch_cal_KxKu
from qpth.qp import QPFunction
def solve_mpc_qpth(H, q, Aeq, beq, G_ieq, h_ieq):
    
    solver = QPFunction(verbose=False)
    solution = solver(H, q, G_ieq, h_ieq, Aeq, beq)
    
    return solution
def cal_qpth_mpc_loss_batch(solution, mpc_config, p_vehicle, data):
    
    batch_size = solution.shape[0]
    kappa = data[:, 0:1]  # kappa是第0列
    vx = data[:, 1:2]     # vx是第1列
    
    # 获取MPC参数
    Q = mpc_config.Q
    R = mpc_config.R
    QN = mpc_config.QN
    Q_diag = torch.diagflat(Q).to(solution.device)
    R_diag = torch.diagflat(R).to(solution.device)
    QN_diag = torch.diagflat(QN).to(solution.device)
    N = mpc_config.N
    Nx = mpc_config.Nx
    Nu = mpc_config.Nu
    decay = mpc_config.decay_factor
    
    Kx, Ku = batch_cal_KxKu(vx, p_vehicle)
    x_ref = (Kx * kappa).to(solution.device)  # (batch_size, Nx)
    u_ref = (Ku * kappa).to(solution.device)  # (batch_size, Nu)

    # 初始化损失张量
    qpth_loss = torch.zeros(batch_size, device=solution.device)
    mpc_loss = torch.zeros(batch_size, device=solution.device)
    
    # 对每个 batch 处理
    for i in range(batch_size):
        solution_i = solution[i, :]  # (Nz,)
        x_ref_i = x_ref[i, :]  # (Nx,)
        u_ref_i = u_ref[i, :]  # (Nu,)
        
        # 从solution中提取状态和控制轨迹
        x_traj = torch.zeros(N+1, Nx, device=solution.device)
        u_traj = torch.zeros(N, Nu, device=solution.device)
        
        for k in range(N):
            x_idx = k * (Nx + Nu)
            u_idx = x_idx + Nx
            x_traj[k, :] = solution_i[x_idx:x_idx+Nx]  
            u_traj[k, :] = solution_i[u_idx:u_idx+Nu]  
        
        # 提取最终状态
        x_traj[N, :] = solution_i[N * (Nx + Nu):N * (Nx + Nu) + Nx]
        
        quadratic_cost = 0
        linear_cost = 0
        constant_cost = 0
        
        # (k=0 to N-1)
        for k in range(N):
            # 当前状态和控制
            wk = decay ** k
            xk = x_traj[k, :]
            uk = u_traj[k, :]
            
            # 二次项: 1/2 * (x^T Q x + u^T R u)
            quadratic_cost += 0.5 * wk * (xk @ Q_diag @ xk + uk @ R_diag @ uk)
            
            # 线性项: -(x^T Q x_ref + u^T R u_ref)
            linear_cost -= wk * (xk @ Q_diag @ x_ref_i + uk @ R_diag @ u_ref_i)
            
            # 常数项: x_ref^T Q x_ref + u_ref^T R u_ref
            constant_cost += wk * (x_ref_i @ Q_diag @ x_ref_i + u_ref_i @ R_diag @ u_ref_i)
        
        # 终端代价 (k=N)
        xN = x_traj[N, :]
        quadratic_cost += 0.5 * (xN @ QN_diag @ xN)
        linear_cost -= xN @ QN_diag @ x_ref_i
        constant_cost += x_ref_i @ QN_diag @ x_ref_i
        
        qpth_loss[i] = quadratic_cost + linear_cost
        
        mpc_loss[i] = qpth_loss[i] * 2 + constant_cost
    
    return qpth_loss, mpc_loss

if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from init_mpc import init_MPC
    from init_vehicle_sim import P_Vehicle
    from init_steer_actuator_sim import init_steer_actuator_sim
    from qpth_solver import solve_mpc_qpth
    from cal_KxKu_batch import batch_cal_KxKu
    from mpc2qp_batch import formulate_qp_problem_batch
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p_vehicle = P_Vehicle(device=device)
    init_steer_actuator_sim(p_vehicle)
    mpc_config = init_MPC(p_vehicle)

    # 数据集，大小为batch_size × 7
    batch_size = 6
    data = torch.zeros(batch_size, 7, device=device)
    data[:, 0] = 0.01   # kappa
    data[:, 1] = 15.0   # vx
    data[:, 2] = 0.01   # vy
    data[:, 3] = 0.01   # r
    data[:, 4] = 0.01   # ey
    data[:, 5] = 0.01  # ephi
    data[:, 6] = 0.01   # delta

    H_batch, q_batch, Aeq_batch, beq_batch, G_ieq_batch, h_ieq_batch = formulate_qp_problem_batch(mpc_config, p_vehicle, data)
    solution = solve_mpc_qpth(H_batch, q_batch, Aeq_batch, beq_batch, G_ieq_batch, h_ieq_batch)
    qpth_loss, mpc_loss = cal_qpth_mpc_loss_batch(solution, mpc_config, p_vehicle, data)
    print(f"qpth损失: {qpth_loss}")
    print(f"qpth_mpc损失: {mpc_loss}")

    # 开始画图，取batch_size=1时
    solution_qpth = solution[0, :] # 纪录决策变量
    kappa = data[:, 0:1]
    vx = data[:, 1:2]
    # 计算参考
    Kx, Ku = batch_cal_KxKu(vx, p_vehicle)
    x_ref = (Kx * kappa)  # (batch_size, Nx)
    u_ref = (Ku * kappa) # (batch_size, Nu)

    x_ref_pqth = x_ref[0, :] # 记录状态量参考
    u_ref_pqth = u_ref[0, :] # 记录控制量参考

    # 转化为numpy
    solution_qpth = solution_qpth.cpu().numpy()
    x_ref_pqth = x_ref_pqth.cpu().numpy()
    u_ref_pqth = u_ref_pqth.cpu().numpy()
    
    # 保存solution_qpth到工作路径
    np.save('solution_qpth.npy', solution_qpth)
    print(f"已保存 solution_qpth 到: solution_qpth.npy")
    
    # MPC参数
    N = mpc_config.N
    Nx = mpc_config.Nx 
    Nu = mpc_config.Nu

    # 开始画图
    x_traj = np.zeros((N+1, Nx))
    u_traj = np.zeros((N, Nu))
    
    # 解析OSQP解
    for k in range(N):
        x_idx = k * (Nx + Nu)
        u_idx = x_idx + Nx
        x_traj[k, :] = solution_qpth[x_idx:x_idx+Nx]
        u_traj[k, :] = solution_qpth[u_idx:u_idx+Nu]
    # 终端状态
    x_traj[N, :] = solution_qpth[N * (Nx + Nu):N * (Nx + Nu) + Nx]
    
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
        ref_value = x_ref_pqth[i]
        ax.axhline(y=ref_value, color='green', linestyle='--', linewidth=2, 
                  label=f'Reference: {ref_value:.4f}')
        
        # QPTH轨迹
        ax.plot(time_steps_x, x_traj[:, i], 'r-o', 
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
    u_ref_value = u_ref_pqth[0]
    ax.axhline(y=u_ref_value, color='green', linestyle='--', linewidth=2, 
              label=f'Reference: {u_ref_value:.4f}')
    
    # QPTH控制轨迹
    ax.plot(time_steps_u, u_traj[:, 0], 'r-o', 
           linewidth=2, markersize=4, label='QPTH Solution')
    
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
    plt.savefig('qpth_mpc_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n图像已保存为 'qpth_mpc_results.png'")
    print("绘图完成！")