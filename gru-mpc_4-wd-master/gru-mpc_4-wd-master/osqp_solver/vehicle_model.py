import numpy as np

def vehicle_model(P_Vehicle, Vx_bar):
    
    Nx = 4 + P_Vehicle.delta_num
    Nu = P_Vehicle.delta_num
    
    # 计算相关参数
    Vx_inv = 1.0 / Vx_bar
    c_vx = Vx_inv * P_Vehicle.Calpha.T  # (4,)
    c_vx_x = c_vx * P_Vehicle.tire_x.T  # (4,)
    c_vx_x_x = c_vx_x * P_Vehicle.tire_x.T  # (4,)
    
    # 初始化A矩阵 (Nx, Nx)
    A = np.zeros((Nx, Nx))
    
    # 计算A矩阵的前4x4部分
    sum_c_vx = np.sum(c_vx)
    sum_c_vx_x = np.sum(c_vx_x)
    sum_c_vx_x_x = np.sum(c_vx_x_x)
    
    # 构造A矩阵的前4行前4列
    A[0, 0] = sum_c_vx / P_Vehicle.m
    A[0, 1] = (sum_c_vx_x / P_Vehicle.m) - Vx_bar
    A[1, 0] = sum_c_vx_x / P_Vehicle.Iz
    A[1, 1] = sum_c_vx_x_x / P_Vehicle.Iz
    A[2, 0] = 1.0
    A[2, 3] = Vx_bar
    A[3, 1] = 1.0
    
    # 计算A_delta相关项
    minus_c_rho = -np.sum(P_Vehicle.Calpha * P_Vehicle.tire_act_delta) / P_Vehicle.m
    minus_c_x_rho = -np.sum(P_Vehicle.Calpha * P_Vehicle.tire_x * P_Vehicle.tire_act_delta) / P_Vehicle.Iz
    
    # 构建A_delta部分
    A_delta = np.zeros((Nx, P_Vehicle.delta_num))
    A_delta[0, :] = minus_c_rho
    A_delta[1, :] = minus_c_x_rho
    A_delta[4:, :] = -np.diag(1.0 / P_Vehicle.t_lag.flatten())
    
    # 将A_delta赋值给A矩阵
    A[:, 4:4+P_Vehicle.delta_num] = A_delta
    
    # 初始化B矩阵 (Nx, Nu)
    B = np.zeros((Nx, Nu))
    B[4:4+P_Vehicle.delta_num, :] = np.diag(1.0 / P_Vehicle.t_lag.flatten())
    
    # E外部扰动矩阵 (Nx, 1)
    E = np.zeros((Nx, 1))
    E[3, 0] = -Vx_bar
    
    return A, B, E

# 使用示例:
if __name__ == '__main__':
    from init_vehicle_sim import P_Vehicle
    from init_steer_actuator_sim import init_steer_actuator_sim
    from mpc2qp import zoh_discrete
    P_Vehicle = P_Vehicle()
    init_steer_actuator_sim(P_Vehicle)
    
    # 标量输入测试
    Vx_bar = 10.0
    A, B, E = vehicle_model(P_Vehicle, Vx_bar)
    Ad, Bd, Ed = zoh_discrete(A, B, E, 0.02)
    print(A)
    print(B)
    print(E)
    print(Ad)
    print(Bd)
    print(Ed)