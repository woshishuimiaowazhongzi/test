import torch
    
def vehicle_model(P_Vehicle, Vx_bar):
    """
    计算车辆状态空间模型矩阵 A, B, E
    
    Args:
        P_Vehicle: VehicleSimulator实例，包含车辆参数
        Vx_bar: 参考纵向速度
        
    Returns:
        A, B, E: 状态空间模型矩阵
    """
    batch_size = Vx_bar.shape[0]
    Nx = 4 + P_Vehicle.delta_num
    Nu = P_Vehicle.delta_num
    #
    # 计算相关参数
    # 4*1 / batch_size*1
    Vx_inv = 1.0/Vx_bar
    c_vx = Vx_inv@P_Vehicle.Calpha.T
    # batch_size*4  1*4
    c_vx_x = c_vx * P_Vehicle.tire_x.T
    # batch_size*4  1*4
    c_vx_x_x = c_vx_x * P_Vehicle.tire_x.T  # 注意这里需要转置以进行矩阵乘法
    
    Nx = 4 + P_Vehicle.delta_num
    
    # 初始化A矩阵，形状为 (batch_size, Nx, Nx)
    A = torch.zeros((batch_size, Nx, Nx), device=P_Vehicle.device)
    
    # 计算A矩阵的前4x4部分
    # batch_size*4 -> batch_size*1
    sum_c_vx = torch.sum(c_vx.squeeze(-1), dim=1, keepdim=True)  # (batch_size, 1)
    sum_c_vx_x = torch.sum(c_vx_x.squeeze(-1), dim=1, keepdim=True)  # (batch_size, 1)
    sum_c_vx_x_x = torch.sum(c_vx_x_x.squeeze(-1), dim=1, keepdim=True)  # (batch_size, 1)
    
    # 构造A矩阵的前4行前4列
    A[:, 0, 0] = sum_c_vx.squeeze() / P_Vehicle.m
    A[:, 0, 1] = (sum_c_vx_x.squeeze() / P_Vehicle.m) - Vx_bar.squeeze()
    A[:, 1, 0] = sum_c_vx_x.squeeze() / P_Vehicle.Iz
    A[:, 1, 1] = sum_c_vx_x_x.squeeze() / P_Vehicle.Iz
    A[:, 2, 0] = 1.0
    A[:, 2, 3] = Vx_bar.squeeze()
    A[:, 3, 1] = 1.0
    # 4*1 * 4*1 / 1*1 -> 1*4 @ 4*1 / 1*1
    minus_c_rho = -P_Vehicle.Calpha.T @ P_Vehicle.tire_act_delta/P_Vehicle.m
    # 4*1 * 4*1 * 4*1 / 1*1 -> (4*1 * 4*1).T @ 4*1 / 1*1
    minus_c_x_rho = -(P_Vehicle.Calpha* P_Vehicle.tire_x).T @ P_Vehicle.tire_act_delta / P_Vehicle.Iz
    # 计算A_delta相关项
    A_delta = torch.cat([
        minus_c_rho,                                    
        minus_c_x_rho,                                  
        torch.zeros(1, P_Vehicle.delta_num, device=P_Vehicle.device),  
        torch.zeros(1, P_Vehicle.delta_num, device=P_Vehicle.device),
        -1.0 * torch.diag(1.0 / P_Vehicle.t_lag.flatten())
    ], dim=0)  
    A[:, :, 4:4+P_Vehicle.delta_num] = A_delta
    # 初始化B矩阵 (batch_size, Nx, Nu)
    B = torch.zeros((batch_size, Nx, Nu), device=P_Vehicle.device)
    B[:, 4:4 + P_Vehicle.delta_num, :] = torch.diag(1.0 / P_Vehicle.t_lag.flatten())
    # E 外部扰动矩阵 (batch_size, Nx, 1)
    E = torch.zeros((batch_size, Nx, 1), device=P_Vehicle.device)
    E[:, 3, 0] = -Vx_bar.squeeze()
    #print(E)
    
    return A, B, E

# 使用示例:
if __name__ == '__main__':
    from init_vehicle_sim import P_Vehicle
    from init_steer_actuator_sim import init_steer_actuator_sim
    P_Vehicle = P_Vehicle(device='cuda' if torch.cuda.is_available() else 'cpu')
    init_steer_actuator_sim(P_Vehicle)
    Vx_bar = 10 * torch.ones(2,1, device=P_Vehicle.device)
    [A, B, E] = vehicle_model(P_Vehicle, Vx_bar)
    print(A)
    print(B)
    print(E)
