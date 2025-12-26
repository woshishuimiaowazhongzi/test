from cal_KxKu import cal_KxKu
from vehicle_model import vehicle_model
import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm

def zoh_discrete(A, B, E, dt):
    
    n = A.shape[0]
    m = B.shape[1]
    p = E.shape[1]
    
    # 构建扩展矩阵 (n + m + p, n + m + p)
    M = np.zeros((n + m + p, n + m + p))
    
    # 填充扩展矩阵
    M[:n, :n] = A
    M[:n, n:n+m] = B
    M[:n, n+m:n+m+p] = E
    
    # 计算矩阵指数
    expMdt = expm(M * dt)
    
    # 提取离散矩阵
    Ad = expMdt[:n, :n]
    Bd = expMdt[:n, n:n+m]
    Ed = expMdt[:n, n+m:n+m+p]
    
    return Ad, Bd, Ed
def formulate_qp_problem(mpc_config, p_vehicle, kappa, vx, x0):

    # 计算连续系统矩阵
    A, B, E = vehicle_model(p_vehicle, vx)
    #print(A)
    # 离散化系统矩阵
    Ad, Bd, Ed = zoh_discrete(A, B, E, mpc_config.Ts)
    #print(Ad)
    
    # 已经是numpy数组，不需要转换
    Ad_np = Ad
    Bd_np = Bd
    Ed_np = Ed
    
    # 计算参考状态和输入
    Kx, Ku = cal_KxKu(vx, p_vehicle)
    x_ref = Kx * kappa
    u_ref = Ku * kappa
    
    x_ref_np = x_ref.flatten()
    u_ref_np = u_ref.flatten()
    
    # MPC参数
    N = mpc_config.N
    Q = mpc_config.Q
    R = mpc_config.R
    QN = mpc_config.QN
    Nx = mpc_config.Nx
    Nu = mpc_config.Nu
    decay = mpc_config.decay_factor
    
    # 总变量数
    Nz = N * (Nx + Nu) + Nx
    
    # 构建H矩阵 (目标函数二次项)
    Q_diag = np.diag(Q.flatten())
    R_diag = np.diag(R.flatten())
    QN_diag = np.diag(QN.flatten())

    # 构建对角块矩阵（加入时间指数衰减，终端不衰减）
    W = sp.block_diag([Q_diag, R_diag])
    blocks = []
    for k in range(N):
        wk = decay ** k
        blocks.append(wk * W)
    # 终端项不衰减
    blocks.append(QN_diag)
    H = sp.block_diag(blocks)
    
    # 构建q向量 (目标函数一次项)
    q_ref = np.zeros(Nz)
    x_u_ref = np.hstack([x_ref_np, u_ref_np])
    x_u_ref_vec = np.tile(x_u_ref, N)
    q_ref[0:Nz - Nx] = x_u_ref_vec
    q_ref[Nz - Nx:Nz] = x_ref_np
    q = -H @ q_ref
    
    # 构建动力学约束矩阵 Aeq * z = beq
    Aeq_dyn = sp.lil_matrix((N * Nx, Nz))
    
    for k in range(N):
        # 当前状态和控制输入在z向量中的位置
        x_idx = k * (Nx + Nu)
        u_idx = x_idx + Nx
        
        # 下一状态在z向量中的位置
        if k < N-1:
            x_next_idx = (k+1) * (Nx + Nu)
        else:
            x_next_idx = N * (Nx + Nu)  # 最后一个状态
        
        # 动力学约束: x_{k+1} = Ad x_k + Bd u_k
        Aeq_dyn[k*Nx:(k+1)*Nx, x_idx:x_idx+Nx] = Ad_np
        Aeq_dyn[k*Nx:(k+1)*Nx, u_idx:u_idx+Nu] = Bd_np
        Aeq_dyn[k*Nx:(k+1)*Nx, x_next_idx:x_next_idx+Nx] = -np.eye(Nx)
    
    # 构建初始状态约束
    Aeq_init = sp.lil_matrix((Nx, Nz))
    Aeq_init[:, :Nx] = np.eye(Nx)
    
    # 合并约束矩阵
    Aeq = sp.vstack([Aeq_dyn, Aeq_init], format='csc')
    
    # 构建约束右侧向量 beq
    beq = np.zeros(N * Nx + Nx)
    
    # 动力学约束右侧 (Ed * kappa)
    kappa_np = kappa
    for k in range(N):
        beq[k*Nx:(k+1)*Nx] = -Ed_np.flatten() * kappa_np
    
    # 初始状态约束右侧
    beq[N*Nx:] = x0
    
    return H, q, Aeq, beq

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
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 负号显示

    vehicle = P_Vehicle(device='cuda' if torch.cuda.is_available() else 'cpu') 
    init_steer_actuator_sim(vehicle) 
    p_vehicle = vehicle.to_numpy() # 把车辆参数转化为numpy版本
    mpc = init_MPC(p_vehicle)
    mpc_config = mpc_config_to_numpy(mpc) # 把mpc初始化参数转化为numpy版本
 
    # 参数
    vx = 15.0
    kappa = 0.01
    
    # 随机生成初始状态
    np.random.seed(42)  # 固定随机种子以便复现
    vy = np.random.rand() * 0  # 横向速度 (m/s)
    r = np.random.rand() * 0   # 横摆角速度 (rad/s)
    ey = np.random.rand() * 0  # 横向位置误差 (m)
    ephi = np.random.rand() * 0  # 航向角误差 (rad)
    
    
    # 组合初始状态向量
    x0 = np.concatenate([np.array([0.01, 0.01, 0.01, 0.01, 0.01])])
    
    # 构建QP问题
    H, q, Aeq, beq = formulate_qp_problem(mpc_config, p_vehicle, kappa, vx, x0)
    
    loss, solution, status = solve_mpc_osqp(H, q, Aeq, beq)
    

    # 验证手动计算的代价函数值
    osqp_loss_manual, mpc_loss_manual = cal_osqp_mpc_loss(solution, mpc_config, p_vehicle, kappa, vx)
    print(f"手动计算osqp损失: {osqp_loss_manual:.6f}")
    print(f"手动计算MPC损失: {mpc_loss_manual:.6f}")
    print(f"osqp损失: {loss:.6f}")
    print(f"差值: {abs(osqp_loss_manual - loss):.8f}")
    



