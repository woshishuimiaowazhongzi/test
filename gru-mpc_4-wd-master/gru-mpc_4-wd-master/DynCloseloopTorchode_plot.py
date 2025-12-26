import torch
import torch.nn as nn
import torchode as to
#from torchdiffeq import odeint
import matplotlib.pyplot as plt
import numpy as np
from policy_net import PolicyNet
from vehicle_dyn import vehicle_dyn
from init_vehicle_sim import P_Vehicle
from init_steer_actuator_sim import init_steer_actuator_sim
import time
from init_mpc import init_MPC
from DynCloseloopWithNet import VehicleDynCloseloop        
if __name__ == '__main__':
    from cal_KxKu_batch import batch_cal_KxKu
    from init_mpc import init_MPC
    
    p_vehicle = P_Vehicle(device='cuda' if torch.cuda.is_available() else 'cpu')
    init_steer_actuator_sim(p_vehicle)
    mpc_config = init_MPC(p_vehicle)


    # 参数设置
    batch_size = 1
    state_dim = 5
    control_dim = 1
    # 时间相关设置
    dt = mpc_config.Ts
    Np = mpc_config.N
    # 车辆参数
    ve_dyn = VehicleDynCloseloop(dt,Np)
    # 初始状态与参数
    p_vehicle = ve_dyn.p_vehicle
    x0 = 0.1*torch.rand(batch_size, state_dim, device=p_vehicle.device)
    kappa = 0.02* torch.ones(batch_size, 1, device=p_vehicle.device)
    Vx_bar = 10.0 * torch.ones(batch_size,1, device=p_vehicle.device)

    Kx, Ku = batch_cal_KxKu(Vx_bar, p_vehicle)
    # 参考 (稳态) 值: Kx, Ku 维度 (batch, Nx) / (batch, Nu)
    # 生成与预测时域长度匹配的参考轨迹 (假设稳态不变)

    

    ve_dyn.set_system_params(kappa, Vx_bar) 
    ve_dyn.load_policy_net('best_model_checkpoint.pth')

    sim_time  =10.0
    Ts = mpc_config.Ts
    sim_steps = int(sim_time / Ts)
    x_traj,u_traj = ve_dyn.solve_N_steps(x0,sim_steps)



    x_ref = (Kx * kappa).unsqueeze(1).repeat(1, sim_steps+1, 1)  # (batch, Np, Nx)
    x_ref_1batch = x_ref[0,:,:].detach().to('cpu').numpy()
    u_ref = (Ku * kappa).unsqueeze(1).repeat(1, sim_steps, 1)  # (batch, Np, Nu)   
    u_ref_1batch = u_ref[0,:,:].detach().to('cpu').numpy()
    x_1batch = x_traj[0,:,:].detach().to('cpu').numpy()
    u_1batch = u_traj[0,:,:].detach().to('cpu').numpy()
    t_eval_x = np.arange(0, sim_steps+1)*mpc_config.Ts
    t_eval_u = np.arange(0, sim_steps)*mpc_config.Ts
    # matplot绘图
            # 绘制状态轨迹 + 参考
    for x_index in range(p_vehicle.Nx):
        plt.subplot(p_vehicle.Nx,1,x_index+1)
        plt.plot(t_eval_x,x_1batch[:,x_index], label=f'x[{x_index}]', color='C0')
        plt.plot(t_eval_x, x_ref_1batch[:,x_index], '--', label=f'x_ref[{x_index}]', color='C1')
        plt.xlabel('Time (s)')
        plt.legend(fontsize=7, loc='best')
    plt.tight_layout()
    plt.show()
    # 绘制输入轨迹
    for u_index in range(p_vehicle.Nu):
        plt.subplot(p_vehicle.Nu,1,u_index+1)
        plt.plot(t_eval_u,u_1batch[:,u_index], label=f'u[{u_index}]', color='C0')
        plt.plot(t_eval_u, u_ref_1batch[:,u_index], '--', label=f'u_ref[{u_index}]', color='C1')
        plt.xlabel('Time (s)')
        plt.legend(fontsize=7, loc='best')
    plt.tight_layout()
    plt.show()
        
