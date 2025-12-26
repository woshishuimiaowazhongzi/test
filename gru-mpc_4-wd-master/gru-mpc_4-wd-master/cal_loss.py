import torch
from cal_KxKu_batch import batch_cal_KxKu
from init_mpc import init_MPC
from DynCloseloopWithNet import VehicleDynCloseloop
from init_mpc import init_MPC
from init_vehicle_sim import P_Vehicle
from init_steer_actuator_sim import init_steer_actuator_sim

def loss_function(mpc_config, x_ref, u_ref, x_N_steps, u_N_steps,with_du=False):
    decay_factor = mpc_config.decay_factor  # 衰减因子，可根据需要调整
    
    # 假设 x_N_steps 形状为 (batch_size, N, state_dim)
    # x_ref 形状需要广播到 (batch_size, N, state_dim)
    
    delta_x = x_N_steps[:,0:mpc_config.N,:] - x_ref.unsqueeze(1) # 确保维度对齐
    delta_u = u_N_steps - u_ref.unsqueeze(1)
    du = u_N_steps[:,0:-1,:]-u_N_steps[:,1:,:]
    # 计算加权平方和 (假设Q, R为对角矩阵，其元素在对角线上)
    cost_state = torch.sum(delta_x**2 * mpc_config.Q.squeeze(), dim=[-2, -1]) # 形状: (batch_size,)
    cost_contorl = torch.sum(delta_u**2 * mpc_config.R.squeeze(), dim=[-2,-1]) # 形状: (batch_size,)    
    if with_du:
        cost_du = torch.sum(du**2 * mpc_config.du_R.squeeze(), dim=[-2, -1]) # 形状: (batch_size,)
    else:
        cost_du = 0.0
    delta_x_terminal = x_N_steps[:, -1, :] - x_ref # 形状: (batch_size, state_dim)
    cost_state_teminal = torch.sum(delta_x_terminal**2 * mpc_config.QN.squeeze(), dim=-1)#batch_size
    #print('terminal cost is:',torch.sum(delta_x_terminal**2 * mpc_config.QN.squeeze(), dim=1))
    cost = cost_state + cost_contorl  + cost_state_teminal + cost_du      
    cost = torch.mean(cost,dim = 0)
    return cost

if __name__ == '__main__':
    p_vehicle = P_Vehicle(device='cuda' if torch.cuda.is_available() else 'cpu')
    init_steer_actuator_sim(p_vehicle)
    mpc_config = init_MPC(p_vehicle)
    N = mpc_config.N
    batch_size = 6
    Nx = 5
    Nu = 1
    vx = 10.0*torch.ones(batch_size,1,device=p_vehicle.device)
    kappa = 0.01*torch.ones(batch_size,1,device=p_vehicle.device)
    Kx, Ku = batch_cal_KxKu(vx, p_vehicle)
    x_ref = Kx * kappa
    u_ref = Ku * kappa

    x_N_steps = torch.zeros(batch_size,N+1,Nx,device=p_vehicle.device)
    u_N_steps = torch.zeros(batch_size,N,Nu,device=p_vehicle.device)
    for i in range(N):
        x_N_steps[:,i,:] = x_ref+0.1
        u_N_steps[:,i,:] = u_ref+0.1
    x_N_steps[:,N,:] = x_ref+0.1
    #delta_x = 0.1 delta_u = 0.1
    # 0.01*5(Nx)*1.0(w)*20(N)+0.01*1(Nu)*5(w)*20 = 1.0+1.0  0.01*5*20 = 1.0 ->3.0
   
    print( loss_function(mpc_config, x_ref,u_ref,x_N_steps, u_N_steps) )
    print(mpc_config.Q.squeeze())