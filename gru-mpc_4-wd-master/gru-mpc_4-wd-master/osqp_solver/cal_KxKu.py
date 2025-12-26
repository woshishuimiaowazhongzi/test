import numpy as np

def cal_KxKu(Vx_bar, P_Vehicle):
    
    # 计算Fy_ref 
    Fy_ref = P_Vehicle.m * Vx_bar**2
    
    # 计算前后轮侧向力
    denominator = P_Vehicle.tire_x[0] - P_Vehicle.tire_x[2]
    Fyf = -P_Vehicle.tire_x[2] / denominator * Fy_ref
    Fyr = P_Vehicle.tire_x[0] / denominator * Fy_ref
    
    # 计算sy_ref
    sy_ref = np.zeros(4)
    sy_ref[0] = Fyf / (P_Vehicle.Calpha[0, 0] + P_Vehicle.Calpha[1, 0])
    sy_ref[1] = sy_ref[0]  # 复制前轮值
    sy_ref[2] = Fyr / (P_Vehicle.Calpha[2, 0] + P_Vehicle.Calpha[3, 0])
    sy_ref[3] = sy_ref[2]  # 复制后轮值
    
    # 初始化参考值
    vy_ref = 0.0
    r_ref = Vx_bar
    e_y = 0.0
    
    # 对于每个轮胎，计算参考转向角
    for tire_index in range(4):
        # 检查是否为非转向轮
        if np.sum(P_Vehicle.tire_act_delta[tire_index]) == 0:
            # vy_ref = Vx_bar * sy_ref - tire_x * r_ref
            vy_ref = Vx_bar * sy_ref[tire_index] - P_Vehicle.tire_x[tire_index, 0] * r_ref
    
    # 计算所有轮胎的参考转向角
    tire_delta_ref = np.zeros(4)
    for tire_index in range(4):
        tire_delta_ref[tire_index] = (
            vy_ref + P_Vehicle.tire_x[tire_index, 0] * r_ref
        ) / Vx_bar - sy_ref[tire_index]
    
    # e_phi = -vy_ref/Vx_bar
    e_phi = -vy_ref / Vx_bar
    
    # 计算Ku (act_tire_delta * tire_delta_ref)
    Ku = P_Vehicle.act_tire_delta @ tire_delta_ref
    vy_ref_array = np.array([vy_ref])
    r_ref_array = np.array([r_ref])
    e_y_array = np.array([e_y])
    e_phi_array = np.array([e_phi])
    
    Kx = np.concatenate((vy_ref_array, r_ref_array, e_y_array, e_phi_array, Ku.flatten()))

    return Kx, Ku

if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import torch
    from init_mpc import init_MPC
    from init_mpc import mpc_config_to_numpy
    from init_vehicle_sim import P_Vehicle
    from init_steer_actuator_sim import init_steer_actuator_sim
    import numpy as np

    vehicle = P_Vehicle(device='cuda' if torch.cuda.is_available() else 'cpu') 
    init_steer_actuator_sim(vehicle) 
    p_vehicle = vehicle.to_numpy() # 把车辆参数转化为numpy版本
    mpc = init_MPC(p_vehicle)
    mpc_config = mpc_config_to_numpy(mpc) # 把mpc初始化参数转化为numpy版本
    # 数据集
    Vx_bar_batch = np.load('result/Vx_bar_batch.npy')

    kappa = 0.01 # 曲率
    
    Kx_list = []
    Ku_list = []

    for i in range(Vx_bar_batch.shape[0]):
        Vx_bar = Vx_bar_batch[i,].item()
        Kx, Ku = cal_KxKu(Vx_bar, p_vehicle)
        Kx_list.append(Kx)
        Ku_list.append(Ku)

    Kx_osqp = np.asarray(Kx_list)
    Ku_osqp = np.asarray(Ku_list)
    
    Kx_net = np.load('result/Kx_batch.npy')
    Ku_net = np.load('result/Ku_batch.npy')

    diff_Kx = Kx_net - Kx_osqp   
    diff_Ku = Ku_net - Ku_osqp

    print(diff_Kx.mean())
    print(diff_Ku.mean())