import torch

def init_MPC(p_vehicle):
    device=p_vehicle.device
    class MPCConfig:
        pass
    mpc_config = MPCConfig()
    mpc_config.Nx = p_vehicle.Nx
    mpc_config.Nu = p_vehicle.Nu
    mpc_config.Nxu = mpc_config.Nx + mpc_config.Nu
    mpc_config.Nd = 1
    mpc_config.N = 120
    mpc_config.Ts = 0.05

    # 状态权重
    w_wy = 1.0
    w_r = 1.0
    w_ey = 1.0
    w_ephi = 1.0
    Qvec = [w_wy, w_r, w_ey, w_ephi] + [1.0] * p_vehicle.delta_num
    mpc_config.Q = torch.tensor(Qvec, device=device)[:,None]
    mpc_config.QN = 1.0 * torch.tensor(Qvec, device=device)[:,None]
    mpc_config.decay_factor = 1.0
    # 控制权重
    delta_num = p_vehicle.delta_num
    R_diag = [5.0] * delta_num
    mpc_config.R = torch.tensor(R_diag, device=device)[:,None]
    du_R_diag = [1.0] * delta_num
    mpc_config.du_R = 1.0 * torch.tensor(du_R_diag, device=device)[:,None]


    # 不等式约束
    mpc_config.N_ineq = 4
    mpc_config.ineq_max = torch.tensor([0.2, 0.2, 0.1, 0.1], device=device)[:,None]
    mpc_config.ineq_min = torch.tensor([-0.2, -0.2, -0.1, -0.1], device=device)[:,None]
    mpc_config.rho = 100.0 * torch.eye(mpc_config.N_ineq, device=device)

    # 最大迭代步数
    mpc_config.max_steps = 100

    return mpc_config
def mpc_config_to_numpy(mpc_config):
    mpc_config_temp = mpc_config
    # 遍历mpc_config所有torch属性并将其转numpy
    for k, v in mpc_config.__dict__.items():
        if isinstance(v, torch.Tensor):
            mpc_config.__dict__[k] = v.cpu().numpy()
    return mpc_config_temp

if __name__ == '__main__':
    from init_vehicle_sim import P_Vehicle
    from init_steer_actuator_sim import init_steer_actuator_sim
    p_vehicle = P_Vehicle(device='cuda' if torch.cuda.is_available() else 'cpu')
    init_steer_actuator_sim(p_vehicle)

    mpc_config = init_MPC(p_vehicle)
    print(mpc_config)

    for attr_name in sorted(dir(mpc_config)):
        # 跳过私有属性和方法
        if not attr_name.startswith('__') and not callable(getattr(mpc_config, attr_name)):
            attr_value = getattr(mpc_config, attr_name)
            print(f"{attr_name}:")
            # 打印属性值
        print(attr_value)

    mpc_config_numpy = mpc_config_to_numpy(mpc_config)
    for attr_name in sorted(dir(mpc_config_numpy)):
        # 跳过私有属性和方法
        if not attr_name.startswith('__') and not callable(getattr(mpc_config_numpy, attr_name)):
            attr_value = getattr(mpc_config_numpy, attr_name)
            print(f"{attr_name}:")
            # 打印属性值
        print(attr_value)    

    