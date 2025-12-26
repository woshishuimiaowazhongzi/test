import torch
import torch.nn as nn
import torchode as to
#from torchdiffeq import odeint
import matplotlib.pyplot as plt
import numpy as np
from policy_net_GRU import PolicyNet
from vehicle_dyn_integrator import VehicleDynIntegrator
from init_vehicle_sim import P_Vehicle
from init_steer_actuator_sim import init_steer_actuator_sim
import time
from init_mpc import init_MPC


class DynCloseloopWithNet():
    def __init__(self,dt=0.02,Np=50):
        self.dt,self.Np = Np,Np
        self.p_vehicle = P_Vehicle(device='cuda' if torch.cuda.is_available() else 'cpu')
        init_steer_actuator_sim(self.p_vehicle)       
        self.policy_net = PolicyNet()
        self.policy_net.to(self.p_vehicle.device)
        self.vehicle_sim = VehicleDynIntegrator(dt)
        

        

    
    def solve_single_step(self, x_k, k):
        batch_size = x_k.shape[0]
        # 控制输入基于当前状态（可根据需要改为基于参考轨迹）
        p_vec = torch.cat([self.kappa, self.Vx_bar, x_k], dim=1)
        u_k = self.policy_net(p_vec)
        self.vehicle_sim.update_u(u_k)
        self.vehicle_sim.update_delta()

        x_k_plus_1 = self.vehicle_sim.solve_single_step(self, x_k)
        return x_k_plus_1, u_k
    def solve(self,x0):
        
        batch_size = x0.shape[0]
        t_eval_samples = self.t_eval_single.unsqueeze(0).repeat(batch_size, 1)
        u_vec = torch.zeros(batch_size, self.Np, self.p_vehicle.Nu, device=self.p_vehicle.device)
        # 单步同步逻辑
        x_traj = torch.zeros(batch_size, self.Np + 1, self.p_vehicle.Nx, device=self.p_vehicle.device)
        x_traj[:,0,:] = x0
        x_k = x0
        for k in range(self.Np):
            x_k1, u_k = self.solve_single_step(x_k, k)
            u_vec[:, k, :] = u_k
            x_traj[:, k+1, :] = x_k1
            x_k = x_k1
        return x_traj, u_vec, t_eval_samples


    def load_policy_net(self,model_path):
        checkpoint = torch.load(model_path, map_location=self.p_vehicle.device)
        # 如果是完整的checkpoint，从中提取模型状态字典
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
  
        # 设置为评估模式
        self.policy_net.eval()
        
if __name__ == '__main__':
    sim_with_net = DynCloseloopWithNet()