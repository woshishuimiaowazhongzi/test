import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cal_KxKu_batch import batch_cal_KxKu
from vehicle_model_batch import vehicle_model
from init_mpc import init_MPC
import numpy as np
import scipy.sparse as sp
import torch

def zoh_discrete_batch(A_batch, B_batch, E_batch, dt):
    batch_size, n, _ = A_batch.shape
    m = B_batch.shape[2]
    
    p = E_batch.shape[2]
    # 构建扩展矩阵批量 (batch_size, n+m+p, n+m+p)
    M_batch = torch.zeros(batch_size, n + m + p, n + m + p, device=A_batch.device)
    
    # 填充扩展矩阵
    M_batch[:, :n, :n] = A_batch
    M_batch[:, :n, n:n+m] = B_batch
    M_batch[:, :n, n+m:n+m+p] = E_batch
    
    # 批量计算矩阵指数
    expMdt_batch = torch.matrix_exp(M_batch * dt)
    
    # 提取离散矩阵
    Ad_batch = expMdt_batch[:, :n, :n]
    Bd_batch = expMdt_batch[:, :n, n:n+m]
    Ed_batch = expMdt_batch[:, :n, n+m:n+m+p]
    
    return Ad_batch, Bd_batch,  Ed_batch

def formulate_qp_problem_batch(mpc_config, p_vehicle, data):
    kappa = data[:, 0:1]
    vx = data[:, 1:2]
    vy = data[:, 2:3]  
    r = data[:, 3:4]
    ey = data[:, 4:5]
    ephi = data[:, 5:6]
    delta = data[:, 6:7]
    A, B, E = vehicle_model(p_vehicle, vx)
    #print(A)
    Ad, Bd, Ed = zoh_discrete_batch(A, B, E, mpc_config.Ts)
    #print(Ad)
    Kx, Ku = batch_cal_KxKu(vx, p_vehicle)
    x_ref = Kx * kappa
    u_ref = Ku * kappa
    N = mpc_config.N
    Q = mpc_config.Q
    R = mpc_config.R
    Nx = mpc_config.Nx
    Nu = mpc_config.Nu
    QN = mpc_config.QN
    Nz = N * (Nx + Nu) + Nx
    decay = mpc_config.decay_factor
    batch_size = Ad.shape[0]
    # 将输出张量放在与 Q 相同的 device 上，避免后续设备不一致
    H = torch.zeros(batch_size, Nz, Nz, device=Q.device)
    q = torch.zeros(batch_size, Nz, device=Q.device)
    Aeq = torch.zeros(batch_size, Nx*N+Nx, Nz, device=Q.device)
    beq = torch.zeros(batch_size, Nx*N+Nx, device=Q.device)
    G_ieq = torch.zeros(batch_size, Nx*N+Nx, Nz, device=Q.device)
    h_ieq = torch.zeros(batch_size, Nx*N+Nx, device=Q.device)
    
    for i in range(batch_size):
        Ad_i = Ad[i, :, :]
        Bd_i = Bd[i, :, :]
        Ed_i = Ed[i, :, :]
        x_ref_i = x_ref[i, :]
        u_ref_i = u_ref[i, :]
        kappa_i = kappa[i, :]

        Q_diag = torch.diagflat(Q)
        R_diag = torch.diagflat(R)
        QN_diag = torch.diagflat(QN)

        W_i = torch.block_diag(Q_diag, R_diag)
        stage_blocks = []
        for k in range(N):
            wk = decay ** k
            stage_blocks.append(wk * W_i)
        H_stage = torch.block_diag(*stage_blocks)
        H_i = torch.block_diag(H_stage, QN_diag).to(Q.device)
        H[i, :] = H_i

        q_ref = torch.zeros(Nz, device=Q.device)
        x_u_ref = torch.hstack([x_ref_i, u_ref_i])
        x_u_ref_vec = x_u_ref.repeat(N)
        q_ref[0:Nz - Nx] = x_u_ref_vec
        q_ref[Nz - Nx:Nz] = x_ref_i
        # 由于 H_i 为对角矩阵，可用对角向量与参考向量逐元素相乘获得 q
        q_i = -H_i.diagonal() * q_ref
        q[i, :] = q_i
        
        
        Aeq_dyn = torch.zeros(N*Nx, Nz, device=Q.device)
        for k in range(N):
            x_idx = k * (Nx + Nu)
            u_idx = x_idx + Nx
            x_next_idx = (k+1) * (Nx + Nu) if k < N-1 else N * (Nx + Nu)
            
            Aeq_dyn[k*Nx:(k+1)*Nx, x_idx:x_idx+Nx] = Ad_i
            Aeq_dyn[k*Nx:(k+1)*Nx, u_idx:u_idx+Nu] = Bd_i
            # 拼接右侧动力学约束右侧 (Ed * kappa)
            beq[i, k*Nx:(k+1)*Nx] = -Ed_i.squeeze() * kappa_i
            if k < N-1:
                Aeq_dyn[k*Nx:(k+1)*Nx, x_next_idx:x_next_idx+Nx] = -torch.eye(Nx)
            else:
                Aeq_dyn[k*Nx:(k+1)*Nx, x_next_idx:x_next_idx+Nx] = -torch.eye(Nx)
        
        Aeq_init = torch.zeros(Nx, Nz, device=Q.device)
        Aeq_init[:, :Nx] = torch.eye(Nx, device=Q.device)
        Aeq_i = torch.cat([Aeq_dyn, Aeq_init], dim=0)
        Aeq[i, :, :] = Aeq_i

        x0 = torch.cat([vy[i], r[i], ey[i], ephi[i], delta[i]])  
        beq[i, N*Nx:] = x0

    return H, q, Aeq, beq, G_ieq, h_ieq


    