import torch
import math

class P_Vehicle:
    def __init__(self, device='cpu'):
        """
        初始化车辆仿真参数        
        Args:
            device (str): 指定计算设备 ('cpu' 或 'cuda')
        """
        self.device = torch.device(device)        
        # 车辆基本参数
        self.miu = torch.tensor(0.85, device=self.device)
        self.m = torch.tensor(1471.4, device=self.device)   # 整车质量
        self.h = torch.tensor(0.54, device=self.device)     # 质心高度
        self.Iz = torch.tensor(1536.7, device=self.device)  # 整车转动惯量
        self.g = torch.tensor(9.8, device=self.device)      # 重力加速度    
        self.radius = torch.tensor(0.305, device=self.device)  #有效滚动半径
        self.J_tire = torch.tensor(0.9, device=self.device)  #有效滚动半径
        self.l = torch.tensor(3.0, device=self.device)
        self.b = torch.tensor(1.35, device=self.device)
        # 轮胎位置 (列向量格式)
        self.tire_x = torch.tensor([[1.015, 1.015, -1.895, -1.895]], device=self.device)
        self.tire_y = torch.tensor([[0.675, -0.675, 0.675, -0.675]], device=self.device)        
        # 轮胎垂向载荷 (来自CarSim的Fy曲线对应Fz，转换为列向量)
        Fz_i = torch.tensor([[4780.74, 4780.74, 1593.58, 1593.58]], device=self.device)        
        # 侧偏角度 (CarSim上Fy对应的侧偏角度，弧度)
        alpha = torch.tensor(10/180*math.pi, device=self.device)        
        # 侧向力 (CarSim上Fy曲线，10deg度侧偏角对应的线性刚度，转换为列向量)
        Fy_i = torch.tensor([[4634.02, 4634.02, 1581.52, 1581.52]], device=self.device)        
        # 静态时的轮胎载荷 (从CarSim读出匀速行驶时车轮的载荷，转换为列向量)
        self.Fz_static = torch.tensor([[4700.0, 4700.0, 2657.0, 2657.0]], device=self.device)        
        # 计算轮胎侧偏刚度
        Calpha_tire = -1.1 * Fy_i / alpha * self.Fz_static / Fz_i
        self.Calpha = Calpha_tire  # 保持列向量格式        
        # 轮胎数量
        self.tire_num = self.tire_x.shape[0]
    #给所有的tensor标量都增加一个batch_size维度,行向量的第零个维度扩展成batch_size维
    def expand_batch_size(self, batch_size):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                # 如果是标量(0维张量)，则添加一个维度变成行向量
                if v.dim() == 0:
                    self.__dict__[k] = v.unsqueeze(0).expand(batch_size, 1)
                else:
                    # 对于已有维度的张量，保持原来的处理方式
                    new_shape = [batch_size] + list(v.shape[1:])
                    self.__dict__[k] = v.expand(*new_shape)
    # 将车辆仿真参数转化成numpy属性
    def to_numpy(self):
        p_vehicle_temp = self
    # 遍历mpc_config所有torch属性并将其转numpy
        for k, v in p_vehicle_temp.__dict__.items():
            if isinstance(v, torch.Tensor):
                p_vehicle_temp.__dict__[k] = v.cpu().numpy()
        return p_vehicle_temp    
        
# 使用示例:

if __name__ == '__main__':
    vehicle = P_Vehicle(device='cuda')
    vehicle.expand_batch_size(2)
    for k, v in vehicle.__dict__.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}")

