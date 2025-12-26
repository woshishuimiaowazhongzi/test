from gen_dataset_utility import  hypercube_grid_sampling_with_noise,hypercube_grid_sampling
from gen_dataset_utility import calc_interval,batch_add_noise
import torch
from calc_vy_r import calc_vy_r
from init_vehicle_sim import P_Vehicle
from init_steer_actuator_sim import init_steer_actuator_sim
from torch.utils.data import DataLoader, TensorDataset
def gen_vehicle_dataset(p_vehicle,num_sample_per_dim,with_noise=False,reserve_boundary=False):
    device = p_vehicle.device
    ay_normal_min,ay_normal_max = -1.0,1.0
    vx_min,vx_max = 1.0,20.0
    alphaf_min,alphaf_max = -0.3,0.3
    alphar_min,alphar_max = -0.3,0.3
    ey_min,ey_max = -0.5,0.5
    ephi_min,ephi_max = -0.5,0.5
    delta_min,delta_max = -0.5,0.5
    bounds=[(ay_normal_min, ay_normal_max),
            (vx_min, vx_max),
            (alphaf_min, alphaf_max),
            (alphar_min, alphar_max),
            (ey_min, ey_max),
            (ephi_min, ephi_max),
            (delta_min, delta_max)]
    if with_noise:
        data = hypercube_grid_sampling_with_noise(num_sample_per_dim, bounds, device = device)
    else:
        data = hypercube_grid_sampling(num_sample_per_dim, bounds, 
                                       reserve_boundary = reserve_boundary, device=device)
    return data,bounds
def create_dataloader(data,batch_size,shuffle=True):
    dataset = TensorDataset(data)    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )    
    return dataloader
#torch tensor:data
def sampling_noise_space_change(data,p_vehicle,interval_list):

    data = batch_add_noise(data,interval_list,p_vehicle.device)
    ay_normal,vx_samples,alpha_f,alpha_r,delta_f = data[:,0],data[:,1],data[:,2],data[:,3],data[:,6]
    ay_max = torch.min(torch.ones_like(ay_normal,device=p_vehicle.device)*8.5,
                       vx_samples*vx_samples*0.2)
    ay_samples = ay_normal*ay_max
    # 通过限制kappa的范围限制vx**2*kappa<=
    vy_samples, r_samples = calc_vy_r(vx_samples, delta_f, alpha_f, alpha_r, p_vehicle)
    kappa_samples = ay_samples/vx_samples/vx_samples
    kappa_samples = torch.clamp(kappa_samples,-0.2,0.2)
    # 如果所有张量都是一维的，可以直接使用 stack
    data_temp = torch.stack([kappa_samples, vx_samples, vy_samples, r_samples,
                            data[:,4], data[:,5], data[:,6]], dim=1)  
    return data_temp

def sampling_space_change(data,p_vehicle):
    ay_normal,vx_samples,alpha_f,alpha_r,delta_f = data[:,0],data[:,1],data[:,2],data[:,3],data[:,6]
    # 通过限制kappa的范围限制vx**2*kappa<=
    ay_max = torch.min(torch.ones_like(ay_normal,device=p_vehicle.device)*8.5,
                       vx_samples*vx_samples*0.2)
    ay_samples = ay_normal*ay_max
    vy_samples, r_samples = calc_vy_r(vx_samples, delta_f, alpha_f, alpha_r, p_vehicle)
    kappa_samples = ay_samples/vx_samples/vx_samples
    kappa_samples = torch.clamp(kappa_samples,-0.2,0.2)
    # 如果所有张量都是一维的，可以直接使用 stack
    data_temp = torch.stack([kappa_samples, vx_samples, vy_samples, r_samples,
                            data[:,4], data[:,5], data[:,6]], dim=1)  
    return data_temp

if __name__ == '__main__':
    p_vehicle = P_Vehicle(device='cuda' if torch.cuda.is_available() else 'cpu')
    init_steer_actuator_sim(p_vehicle)
    num_sample_per_dim = [5,5,3,3,3,3,3]
    ############ 生成训练数据集
    data_org,bounds_org = gen_vehicle_dataset(p_vehicle,num_sample_per_dim,with_noise=False,reserve_boundary=True)
    #dara_org加入噪声
    interval_list = calc_interval(num_sample_per_dim,bounds_org,reserve_boundary=True)

    #dara_org加入噪声

    print(interval_list)
    data_change_noise = sampling_noise_space_change(data_org,p_vehicle,interval_list)

    
    #绘制data_org和data_change前两维状态
    from matplotlib import pyplot as plt
    plt.subplot(1,2,1)
    plt.scatter(data_org.detach().cpu().numpy()[:,0],data_org.detach().cpu().numpy()[:,1])
   # plt.subplot(2,2,2)
   # plt.scatter(data_change.detach().cpu().numpy()[:,0],data_change.detach().cpu().numpy()[:,1])   
    plt.subplot(1,2,2)
    plt.scatter(data_change_noise.detach().cpu().numpy()[:,0],data_change_noise.detach().cpu().numpy()[:,1])  
    plt.show()
    plt.figure()
    dataloader = create_dataloader(data_change_noise,batch_size=128)
    ############ 生成测试数据集
    dev_org,bounds_temp=gen_vehicle_dataset(p_vehicle,num_sample_per_dim,with_noise=True,reserve_boundary=False)
    dev_change = sampling_space_change(dev_org,p_vehicle)
    # dev_data
    dev_loader = create_dataloader(dev_change,batch_size=128)
    #绘图
    plt.subplot(1,2,1)
    plt.scatter(dev_org.detach().cpu().numpy()[:,0],dev_org.detach().cpu().numpy()[:,1])
    plt.subplot(1,2,2)
    plt.scatter(dev_change.detach().cpu().numpy()[:,0],dev_change.detach().cpu().numpy()[:,1])
    plt.show()