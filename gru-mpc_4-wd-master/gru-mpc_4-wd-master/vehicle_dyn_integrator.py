import torch
import torch.nn as nn
import torchode as to
#from torchdiffeq import odeint
import matplotlib.pyplot as plt
import numpy as np
from policy_net import PolicyNet
from vehicle_dyn import vehicle_dyn,calc_sx
from vehicle_dyn import calc_sx
from init_vehicle_sim import P_Vehicle
from init_steer_actuator_sim import init_steer_actuator_sim
import time
from init_mpc import init_MPC
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/tire_model')
from tire_model.Pac02Tire import Pac02Tire

class VehicleDynIntegrator():
    def __init__(self,dt=0.002):
        self.p_vehicle = P_Vehicle(device='cuda' if torch.cuda.is_available() else 'cpu')
        init_steer_actuator_sim(self.p_vehicle)    
        self.tire_model = Pac02Tire("tire_model/HMMWV_Pac02Tire.json", device=p_vehicle.device)   
        self.Vx_bar = None
        self.kappa = None
        self.dt = dt
        self.u_vec = None
        # 单步求解器：使用固定控制 u_k，在 [t_k, t_{k+1}] 上积分
        # 通过 self.current_u 注入每步常量控制
        term_step = to.ODETerm(self._ode_fixed_u)
        step_method_step = to.Dopri5(term=term_step)
        controller_step = to.IntegralController(atol=1e-6, rtol=1e-5, term=term_step)
        self.step_solver = torch.compile(to.AutoDiffAdjoint(step_method_step, controller_step))
        self.current_u = None  # (batch, Nu) 每步固定控制

        
    def set_system_params(self, kappa, Vx_bar):
        """动态设置系统参数"""
        self.kappa = kappa
        self.Vx_bar = Vx_bar
        

    def _ode_fixed_u(self, t, x):
        return vehicle_dyn(self.p_vehicle, x, self.current_u, self.delta,self.tire_model)   
            
    def update_u(self, u):
        """更新控制"""
        self.current_u = u
    def update_delta(self, delta):
        self.delta = delta

    def solve_single_step(self, x_k):
        batch_size = x_k.shape[0]
        # 构造该步的统一时间网格 (所有 batch 同一 t_k, t_{k+1})
        t_k_val = 0.0
        t_k1_val = self.dt
        t_eval = torch.stack([
            torch.full((batch_size,), t_k_val, device=self.p_vehicle.device),
            torch.full((batch_size,), t_k1_val, device=self.p_vehicle.device)
        ], dim=1)
        problem = to.InitialValueProblem(y0=x_k, t_eval=t_eval)
        sol = self.step_solver.solve(problem)
        # sol.ys 形状: (batch, Np, Nx)
        x_k1 = sol.ys[:,-1,:]
        return x_k1

    def solve_N_steps(self, x0,N):
        batch_size = x0.shape[0]
        x_traj = torch.zeros(batch_size, N+1, self.p_vehicle.Nx, device=self.p_vehicle.device)
        x_traj[:,0,:] = x0
        x_k = x0
        for k in range(N):
            x_k1 = self.solve_single_step(x_k)
            x_traj[:, k+1, :] = x_k1
            x_k = x_k1
        return x_traj
    

        
if __name__ == '__main__':
    from cal_KxKu_batch import batch_cal_KxKu
    from init_mpc import init_MPC
    
    p_vehicle = P_Vehicle(device='cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    p_vehicle.expand_batch_size(batch_size)
    init_steer_actuator_sim(p_vehicle)
    mpc_config = init_MPC(p_vehicle)

    # 时间相关设置
    dt = 0.0001

    # 车辆参数
    ve_sim = VehicleDynIntegrator(dt)
    # 初始状态与参数
    p_vehicle = ve_sim.p_vehicle
    x0 = 0.0*torch.rand(batch_size, p_vehicle.Nx, device=p_vehicle.device)

    u = 1.0 * torch.ones(batch_size, p_vehicle.Nu, device=p_vehicle.device)
    delta = 0.02* torch.ones(batch_size, 1, device=p_vehicle.device)

    x0[:, 0:1] = 2.0*torch.ones(batch_size, 1, device=p_vehicle.device) # vx
    x0[:, 3:3+p_vehicle.tire_num] =2.0*torch.ones(batch_size, p_vehicle.tire_num, device=p_vehicle.device)# vx_tire

    ve_sim.update_u(u)
    ve_sim.update_delta(delta)
    Np = 200
    x_traj = ve_sim.solve_N_steps(x0,Np)
    x_traj_batch0 = x_traj[0,:,:]
    ## 计算轮胎滑移率
    sx_traj = torch.zeros(batch_size,Np, p_vehicle.tire_num,device=p_vehicle.device)
    for k in range(Np):
        sx_traj[:,k,:] = calc_sx(p_vehicle,x_traj[:,k,:])
    sx_traj_batch0 = sx_traj[0,:,:].detach().cpu().numpy()

    #将x_traj_batch0转到cpu，并转换为numpy
    x_traj_batch0 = x_traj_batch0.detach().cpu().numpy()
    #使用matplotlib绘制所有的状态子图
    for i in range(p_vehicle.Nx):
        plt.subplot(p_vehicle.Nx, 1, i+1)
        plt.plot(x_traj_batch0[:,i])
        plt.ylabel('x'+str(i))
        plt.grid(True)
    plt.show()

    #绘制sx_traj_batch0轨迹
    for i in range(p_vehicle.tire_num):
        plt.subplot(p_vehicle.tire_num, 1, i+1)
        plt.plot(sx_traj_batch0[:,i])
        plt.ylabel('sx'+str(i))
        plt.grid(True)
    plt.show()

    

    

