import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
def hypercube_grid_sampling(num_samples_per_dim, bounds, reserve_boundary=False, device='cpu'):
    """
    使用网格采样生成超立方体样本
    
    Args:
        num_samples_per_dim: 每个维度的样本数
        bounds: 每个维度的边界 [(min1, max1), (min2, max2), ...]
        reserve_boundary: 是否在边界预留一半采样间隔,默认为False
        device: 设备
    
    Returns:
        samples: 形状为 (num_samples_total, num_dimensions) 的张量
    """
    # 为每个维度生成线性样本
    grids = []
    dim_index = 0
    for min_val, max_val in bounds:
        if reserve_boundary:
            # 计算采样间隔
            interval = (max_val - min_val) / num_samples_per_dim[dim_index] if num_samples_per_dim[dim_index] > 1 else 0
            # 预留一半间隔，调整起始和结束位置
            adjusted_min = min_val + interval / 2
            adjusted_max = max_val - interval / 2
            grid = torch.linspace(adjusted_min, adjusted_max, num_samples_per_dim[dim_index], device=device)
        else:
            grid = torch.linspace(min_val, max_val, num_samples_per_dim[dim_index], device=device)
        
        dim_index += 1
        grids.append(grid)
    
    # 创建网格
    mesh_grids = torch.meshgrid(*grids, indexing='ij')    
    # 展平并组合
    samples = torch.stack([grid.flatten() for grid in mesh_grids], dim=1)
    
    return samples
def hypercube_grid_sampling_with_noise(num_samples_per_dim, bounds, device='cpu'):
    """
    使用网格采样生成超立方体样本，并在每个点的每个维度加入均匀噪声
    """
    grids = []
    intervals = []

    for dim, (min_val, max_val) in enumerate(bounds):
        n_samples = num_samples_per_dim[dim]
        interval = (max_val - min_val) / n_samples if num_samples_per_dim[dim]>1 else 0
        intervals.append(interval)

        adjusted_min = min_val + interval / 2
        adjusted_max = max_val - interval / 2

        grid = torch.linspace(adjusted_min, adjusted_max, n_samples, device=device)
        grids.append(grid)

    mesh_grids = torch.meshgrid(*grids, indexing='ij')
    samples = torch.stack([grid.flatten() for grid in mesh_grids], dim=1)

    # 使用 uniform_() 方法生成噪声
    intervals_tensor = torch.tensor(intervals, device=device)

    
    # 创建未初始化的噪声张量，然后使用 uniform_() 填充
    noise = torch.empty(samples.shape, device=device)
    noise.uniform_(-0.5, 0.5)  # 生成 [-0.5, 0.5) 的均匀分布
    
    # 根据每个维度的间隔进行缩放
    noise = noise * intervals_tensor
    
    samples = samples + noise
    
    return samples
 
def calc_interval(num_samples_per_dim,bounds,reserve_boundary=True):
    num_dimensions = len(bounds)
    interval_list = []    
    for dim in range(num_dimensions):
        # 计算该维度的网格间隔
        min_val, max_val = bounds[dim]
        if reserve_boundary == False:
            interval = (max_val - min_val) / (num_samples_per_dim[dim] - 1) if num_samples_per_dim[dim] > 1 else 0
        else:
            interval = (max_val - min_val) / num_samples_per_dim[dim] if num_samples_per_dim[dim] > 1 else 0 
        interval_list.append(interval)
    return interval_list
def batch_add_noise(samples,interval_list,device='cpu'):
     # 添加噪声（除了边界点）
    noisy_samples = samples.clone()
    
        # 使用 uniform_() 方法生成噪声
    intervals_tensor = torch.tensor(interval_list, device=device)

    
    # 创建未初始化的噪声张量，然后使用 uniform_() 填充
    noise = torch.empty(samples.shape, device=device)
    noise.uniform_(-0.5, 0.5)  # 生成 [-0.5, 0.5) 的均匀分布
    
    # 根据每个维度的间隔进行缩放
    noise = noise * intervals_tensor
    
    noisy_samples = noisy_samples + noise
    return noisy_samples

    
    



if __name__ == '__main__':
    samples_per_dim = np.array([5,5,5,5,5,5])
    bounds = [(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)]
    #设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    samples = hypercube_grid_sampling(samples_per_dim, bounds, reserve_boundary=True, device=device)
    noise_ratio = 0.5
    samples_noise = hypercube_grid_sampling_with_noise(samples_per_dim, bounds, device=device)
    print(samples.shape)
    print(samples_noise.shape)

    interval_list= calc_interval(samples_per_dim,bounds)
    print('interval is:',interval_list)

    samples_add_noise = batch_add_noise(samples,interval_list,device)
    #matplotlib绘制
    from matplotlib import pyplot as plt
    #子图绘制
    plt.subplot(1,3,1)
    plt.scatter(samples.detach().cpu().numpy()[:,0],samples.detach().cpu().numpy()[:,1])
    plt.subplot(1,3,2)
    plt.scatter(samples_noise.detach().cpu().numpy()[:,1],
                samples_noise.detach().cpu().numpy()[:,2])
    plt.subplot(1,3,3)
    plt.scatter(samples_add_noise.detach().cpu().numpy()[:,1],
                samples_add_noise.detach().cpu().numpy()[:,2])
    plt.show()
    print(samples)


