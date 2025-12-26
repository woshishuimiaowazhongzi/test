import torch

def batch_cal_KxKu(Vx_bar, P_Vehicle):
    """
    批量计算Kx和Ku - 并行版本
    
    参数:
    Vx_bar: 参考速度批量 (batch_size,)
    P_Vehicle: 车辆参数批量字典或结构体
    
    返回:
    Kx: 状态参考值批量 (batch_size, 5)
    Ku: 输入参考值批量 (batch_size, n_actuators)
    """
    batch_size = Vx_bar.shape[0]
    
    # 从参数批量中提取所需参数
    # 计算Fy_ref 
    Fy_ref = P_Vehicle.m * Vx_bar**2  # (batch_size,)
    
    # 计算前后轮侧向力
    Fyf = -P_Vehicle.tire_x[2]/(P_Vehicle.tire_x[0]-P_Vehicle.tire_x[2])*Fy_ref
    Fyr = P_Vehicle.tire_x[0]/(P_Vehicle.tire_x[0]-P_Vehicle.tire_x[2])*Fy_ref
    denominator = P_Vehicle.tire_x[0] - P_Vehicle.tire_x[2]  # (batch_size,)
    Fyf = -P_Vehicle.tire_x[2] / denominator * Fy_ref  # (batch_size,)
    Fyr = P_Vehicle.tire_x[0] / denominator * Fy_ref  # (batch_size,)
    
    # 计算sy_ref
    sy_ref = torch.zeros(batch_size, 4, device=Vx_bar.device)
    # 保持列形状为 (batch,1)，避免后续与 (batch,1) 相乘时发生 (batch,batch) 的广播
    sy_ref[:, 0:1] = Fyf / (P_Vehicle.Calpha[0] + P_Vehicle.Calpha[1])
    sy_ref[:, 1:2] = sy_ref[:, 0:1]  # 复制前轮值
    sy_ref[:, 2:3] = Fyr / (P_Vehicle.Calpha[2] + P_Vehicle.Calpha[3])
    sy_ref[:, 3:4] = sy_ref[:, 2:3]  # 复制后轮值
    
    # 初始化参考值
    vy_ref = torch.zeros(batch_size, 1, device=Vx_bar.device)
    r_ref = Vx_bar.clone()  # (batch_size,)
    e_y = torch.zeros(batch_size,1, device=Vx_bar.device)
    
    
    # 对于每个轮胎，计算参考转向角
    for tire_index in range(4):
        # tire_delta_ref = (vy_ref + tire_x * r_ref)/Vx_bar - sy_ref
        # 这里需要先确定vy_ref，使用非转向轮的信息
        
        # 检查是否为非转向轮
       
        
        if torch.sum(P_Vehicle.tire_act_delta[tire_index]) == 0:
            # vy_ref = Vx_bar * sy_ref - tire_x * r_ref
            # 使用列向量 (batch,1) 逐样本相乘，避免广播成 (batch,batch)
            vy_ref = Vx_bar * sy_ref[:, tire_index:tire_index+1] - P_Vehicle.tire_x[tire_index] * r_ref
    # print(vy_ref.size())  # 注释掉调试打印
    # 重新计算所有轮胎的参考转向角
    # for tire_index in range(4):
    #     tire_delta_ref[:, tire_index] = (
    #         vy_ref + P_Vehicle.tire_x[tire_index] * r_ref
    #     ) / Vx_bar - sy_ref[:, tire_index]
    # batch_size*1 1*4
    tire_delta_ref = (
            vy_ref + r_ref@P_Vehicle.tire_x.T 
        ) / Vx_bar - sy_ref
    
    # e_phi = -vy_ref/Vx_bar
    e_phi = -vy_ref / Vx_bar
    
    # 计算Ku (act_tire_delta * tire_delta_ref)
    # 假设act_tire_delta是固定的矩阵 (n_actuators, tire_num)
    
    # ndelta* 4   batch_size * 4 -> batch_size * ndelta
    Ku = (P_Vehicle.act_tire_delta @ tire_delta_ref.T).T

    Kx = torch.cat((vy_ref, r_ref,e_y,e_phi,Ku), dim=1)  # (batch_size, 5)
    
    return Kx, Ku

if __name__ == '__main__':
    from init_mpc import init_MPC
    import numpy as np
    from init_vehicle_sim import P_Vehicle
    from init_steer_actuator_sim import init_steer_actuator_sim

    p_vehicle = P_Vehicle(device='cuda' if torch.cuda.is_available() else 'cpu')
    init_steer_actuator_sim(p_vehicle)       
    mpc_config = init_MPC(p_vehicle)
    
    batch_size = 100

    Vx_bar = 10 * torch.rand(batch_size, 1, device=p_vehicle.device)+1.0
    kappa = 0.05 * torch.rand(batch_size, 1, device=p_vehicle.device)
    # 保存Vx_bar到result目录下
    np.save('result/Vx_bar_batch.npy', Vx_bar.cpu().numpy())
    np.save('result/kappa_batch.npy', kappa.cpu().numpy())
    Kx, Ku = batch_cal_KxKu(Vx_bar, p_vehicle)
    
    # 保存Kx和Ku到result目录下
    np.save('result/Kx_batch.npy', Kx.cpu().numpy())
    np.save('result/Ku_batch.npy', Ku.cpu().numpy())
    x_ref = Kx * kappa
    u_ref = Ku * kappa