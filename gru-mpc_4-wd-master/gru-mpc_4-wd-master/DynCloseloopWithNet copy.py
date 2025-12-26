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

class DynCloseloopWithNet():
    def __init__(self,dt=0.02,Np=50):
        self.p_vehicle = P_Vehicle(device='cuda' if torch.cuda.is_available() else 'cpu')
        init_steer_actuator_sim(self.p_vehicle)       
        self.policy_net = PolicyNet()
        self.policy_net.to(self.p_vehicle.device)
        self.Vx_bar = None
        self.kappa = None
        self.dt = dt
        self.Np = Np
        self.t_eval_single = torch.linspace(0, Np*dt, Np+1,device=self.p_vehicle.device)
        self.u_vec = None
        # 单步求解器：使用固定控制 u_k，在 [t_k, t_{k+1}] 上积分
        # 通过 self.current_u 注入每步常量控制
        term_step = to.ODETerm(self._ode_fixed_u)
        step_method_step = to.Dopri5(term=term_step)
        controller_step = to.IntegralController(atol=1e-6, rtol=1e-3, term=term_step)
        self.step_solver = torch.compile(to.AutoDiffAdjoint(step_method_step, controller_step))
        self.current_u = None  # (batch, Nu) 每步固定控制

        

        

    
    def solve_single_step(self, x_k, k):
        batch_size = x_k.shape[0]
        # 控制输入基于当前状态（可根据需要改为基于参考轨迹）
        p_vec = torch.cat([self.kappa, self.Vx_bar, x_k], dim=1)
        u_k = self.policy_net(p_vec)
        self.update_u(u_k)
        # 构造该步的统一时间网格 (所有 batch 同一 t_k, t_{k+1})
        t_k_val = self.t_eval_single[k]
        t_k1_val = self.t_eval_single[k+1]
        t_eval = torch.stack([
            torch.full((batch_size,), t_k_val, device=self.p_vehicle.device),
            torch.full((batch_size,), t_k1_val, device=self.p_vehicle.device)
        ], dim=1)
        problem = to.InitialValueProblem(y0=x_k, t_eval=t_eval)
        sol = self.step_solver.solve(problem)
        # sol.ys 形状: (batch, Np, Nx)
        x_k_plus_1 = sol.ys[:,-1,:]
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
    from cal_KxKu_batch import batch_cal_KxKu
    from init_mpc import init_MPC
    
    p_vehicle = P_Vehicle(device='cuda' if torch.cuda.is_available() else 'cpu')
    init_steer_actuator_sim(p_vehicle)
    mpc_config = init_MPC(p_vehicle)


    # 参数设置
    batch_size = 1000000 # 160*0.05 // 
    state_dim = 5
    control_dim = 1
    # 时间相关设置
    dt = mpc_config.Ts
    Np = mpc_config.N
    # 车辆参数
    ve_dyn = VehicleDynCloseloop(dt,Np)
    # 初始状态与参数
    p_vehicle = ve_dyn.p_vehicle
    x0 = 0.01*torch.rand(batch_size, state_dim, device=p_vehicle.device)
    kappa = 0.01* torch.ones(batch_size, 1, device=p_vehicle.device)
    Vx_bar = 10.0 * torch.ones(batch_size,1, device=p_vehicle.device)


    ve_dyn.set_system_params(kappa, Vx_bar) 
    #ve_dyn.load_policy_net('best_model_checkpoint.pth')
    # 特化state_func
    # u_test = 0.1*torch.ones(batch_size,control_dim, device=p_vehicle.device)

    # 统计求解时间
    with torch.no_grad():
        for i in range(10):
            start_time = time.time()
            x_batch,u_batch,t_batch = ve_dyn.solve(x0) 
            end_time = time.time()
            print('time cost:',end_time-start_time)
    
    # x = x_batch.detach().cpu().numpy()
    # u = u_batch.detach().cpu().numpy()
    # t = t_batch.detach().cpu().numpy()

    # for x_index in range(p_vehicle.Nx):
    #     plt.subplot(p_vehicle.Nx,1,x_index+1)        
    #     plt.plot(t[0,:],x[0,:,x_index], label='x0_'+str(x_index))
    #     plt.legend()
    # plt.show()

    # plt.plot(t[0,:-1],u[0,:,0])
    # plt.show()

    # u_net = u[0, :, :]
    # x_net = x[0, :, :]
    # np.save('result/u_net.npy', u_net)
    #np.save('result/x_net.npy', x_net)
    #print(u_net.shape)
    #final_state = sol.ys[-1] # 获取最终时间点的状态
    #print(final_state)