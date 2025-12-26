from init_vehicle_sim import P_Vehicle
from init_steer_actuator_sim import init_steer_actuator_sim
import torch

P_Vehicle = P_Vehicle(device='cuda' if torch.cuda.is_available() else 'cpu')
init_steer_actuator_sim(P_Vehicle)

def calc_vy_r(vx,delta,alpha_f,alpha_r,P_Vehicle):

    L = P_Vehicle.tire_x[0] - P_Vehicle.tire_x[3]
    r = vx * (alpha_f + delta - alpha_r) / L
    vy = vx * (P_Vehicle.tire_x[0] * alpha_r - P_Vehicle.tire_x[3] * alpha_f - P_Vehicle.tire_x[3] * delta) / L
    return vy, r

if __name__ == '__main__':
    # batch_size = 20
    # Vx_bar = 10*torch.ones(3, 1, device=P_Vehicle.device)
    # delta = 1.57/180*3.14*torch.ones(3, 1, device=P_Vehicle.device)
    # alpha_f = -0.31/180*3.14*torch.ones(3, 1, device=P_Vehicle.device)
    # alpha_r = -0.29/180*3.14*torch.ones(3, 1, device=P_Vehicle.device)

    # alpha_f_min, alpha_f_max = -0.07, 0.07
    # alpha_r_min, alpha_r_max = -0.07, 0.07
    # delta_f_min, delta_f_max = -0.6, 0.6
    # alpha_f_samples = torch.FloatTensor(batch_size, 1).uniform_(alpha_f_min, alpha_f_max).to(device = 'cuda')
    # alpha_r_samples = torch.FloatTensor(batch_size, 1).uniform_(alpha_r_min, alpha_r_max).to(device = 'cuda')
    # delta_f_samples = torch.FloatTensor(batch_size, 1).uniform_(delta_f_min, delta_f_max).to(device = 'cuda')
    # vx_samples = 10*torch.ones(batch_size,1).to(device = 'cuda')

    # vy, r = calc_vy_r(vx_samples,delta_f_samples,alpha_f_samples,alpha_r_samples,P_Vehicle)
    #vy, r = calc_vy_r(Vx_bar,delta,alpha_f,alpha_r,P_Vehicle)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from init_vehicle_sim import P_Vehicle as P_Vehicle_class
    P_Vehicle_instance = P_Vehicle_class(device=device)
    init_steer_actuator_sim(P_Vehicle_instance)
    num_sample = 10

    vx_min ,vx_max = 0.001, 15.0
    vx_sampled = 10 # 定车速
    kappa_min, kappa_max = -0.2, 0.2
    kappa_sampled = 0.03 # 定曲率
    vx_samples = torch.FloatTensor(num_sample, 1).uniform_(vx_min, vx_max).to(device)
    # print(vx_samples)
    kappa_lower_bound1 = torch.full_like(vx_samples, kappa_min)
    kappa_lower_bound2 = -8.5 / (vx_samples ** 2)
    kappa_upper_bound1 = torch.full_like(vx_samples, kappa_max)
    kappa_upper_bound2 = 8.5 / (vx_samples ** 2)
    
    kappa_min_values = torch.max(kappa_lower_bound1, kappa_lower_bound2)
    kappa_max_values = torch.min(kappa_upper_bound1, kappa_upper_bound2)
    # print(kappa_min_values)
    # print(kappa_max_values)
    # 为每个样本生成在对应范围内的kappa值
    kappa_samples = torch.empty_like(vx_samples).uniform_(0, 1)

    kappa_samples = kappa_min_values + (kappa_max_values - kappa_min_values) * kappa_samples
    # print(kappa_samples)

    alpha_f_min, alpha_f_max = -0.14, 0.14
    alpha_r_min, alpha_r_max = -0.14, 0.14
    delta_f_min, delta_f_max = -0.35, 0.35
    ey_min, ey_max = -0.3, 0.3
    ephi_min, ephi_max = -0.1, 0.1
    
    alpha_f = torch.FloatTensor(num_sample, 1).uniform_(alpha_f_min, alpha_f_max).to(device)
    alpha_r = torch.FloatTensor(num_sample, 1).uniform_(alpha_r_min, alpha_r_max).to(device)
    delta_f = torch.FloatTensor(num_sample, 1).uniform_(delta_f_min, delta_f_max).to(device)
    
    # 从侧偏角和转向角计算vy和r
    # vy_samples, r_samples = calc_vy_r(vx_samples, delta_f, alpha_f, alpha_r, P_Vehicle_instance) #变车速
    vy_samples, r_samples = calc_vy_r(vx_sampled, delta_f, alpha_f, alpha_r, P_Vehicle_instance)    
    print(vy_samples)  # 注释掉调试打印
    print(vy_samples.size())
    print(r_samples)
    print(r_samples.size())
    print(delta_f)
    print(alpha_f)
    print(alpha_r)