import torch
import torch.nn as nn
from load_transfer import load_transfer
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/tire_model')
from tire_model.Pac02Tire import Pac02Tire


def vehicle_dyn(p_vehicle,x,u,delta,tire_model): 
    vx,vy,r = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    vx_tire = x[:, 3:3+p_vehicle.tire_num]
    acc_motor = u[:, 0:p_vehicle.drive_num]
    # 轮胎模型
    kappa = (vx-vx_tire)/torch.max(vx_tire,vx)
    alpha = (vy + p_vehicle.tire_x*r)/vx - p_vehicle.tire_act_delta.T * delta
    Fz = load_transfer(torch.sum(acc_motor,dim=-1,keepdim=True),vx*r,p_vehicle)
    results = tire_model.forward(kappa,alpha, Fz, omega=0)
    Fx = -results['fx']
    Fy = -results['fy']
    #Fx = s_x * p_vehicle.Calpha 
    #Fy = s_y * p_vehicle.Calpha # 侧偏角乘以轮胎刚度
    # 0.纵向速度状态
    sum_Fx = torch.sum(Fx, dim=1, keepdim=True)
    vx_dot = (sum_Fx / p_vehicle.m)+(r*vy)
    # 1.侧向速度一阶导数
    sum_Fy = torch.sum(Fy, dim=1, keepdim=True)
    vy_dot = (sum_Fy / p_vehicle.m) - (r*vx) 
    # 2.横摆角速度一阶导数
    r_dot1 = (p_vehicle.tire_x * Fy)/p_vehicle.Iz
    r_dot = torch.sum(r_dot1, dim=1, keepdim=True)
    # 3. 轮速状态
    vx_tire_dot = (acc_motor*p_vehicle.m-Fx)*p_vehicle.radius/p_vehicle.J_tire
    # 导数的组合
    dx_dt = torch.cat([vx_dot,vy_dot,r_dot,vx_tire_dot], dim=1)
    return dx_dt
def calc_sx(p_vehicle,x):
    vx = x[:, 0:1]
    vx_tire = x[:, 3:3+p_vehicle.tire_num]
    s_x = (vx-vx_tire)/torch.max(vx_tire,vx)
    return s_x

# 使用示例:
if __name__ == '__main__':
    from init_vehicle_sim import P_Vehicle
    from init_steer_actuator_sim import init_steer_actuator_sim
    p_vehicle = P_Vehicle(device='cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 3
    p_vehicle.expand_batch_size(batch_size)
    init_steer_actuator_sim(p_vehicle)
    

    
    x = 0.002*torch.ones(batch_size, p_vehicle.Nx, device=p_vehicle.device)
    x[:, 0:1] = 10.0*torch.ones(batch_size, 1, device=p_vehicle.device) # vx
    x[:, 3:3+p_vehicle.tire_num] = 10.5*torch.ones(batch_size, p_vehicle.tire_num, device=p_vehicle.device)# vx_tire

    u = 1.0*torch.ones(batch_size, p_vehicle.Nu,device=p_vehicle.device)
    delta = 0.02* torch.ones(batch_size, 1, device=p_vehicle.device)
    tire_model = Pac02Tire("tire_model/HMMWV_Pac02Tire.json", device=p_vehicle.device)
    dx_dt = vehicle_dyn(p_vehicle,x,u, delta,tire_model)
    print(dx_dt)
    #solution_zoh = odeint(ode_system_zoh, x0, t_eval, method='implicit_adams', rtol=1e-3, atol=1e-6)  # 可以在 options 中调整 SciPy 求解器的参数')

