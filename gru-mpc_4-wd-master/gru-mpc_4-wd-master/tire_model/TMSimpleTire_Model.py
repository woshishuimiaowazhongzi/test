import torch
import numpy as np
import json
import os
import math

# 导入父类 (确保当前目录下有这些文件)
from ChForceElementTire import ChForceElementTire
from ChPart import ChPart

class TMsimpleTire(ChForceElementTire):
    """
    基于 Project Chrono TMsimpleTire 的 PyTorch 实现。
    继承结构: TMsimpleTire -> ChForceElementTire -> ChTire -> ChPart
    """
    def __init__(self, json_filename, device='cpu'):
        # 调用父类构造函数初始化 name 和 device
        super().__init__("TMsimpleTire", device)
        
        # --- 初始化默认状态变量 ---
        self.mu_0 = torch.tensor(0.8, device=self.device)
        self.d1 = torch.tensor(0.0, device=self.device) # 垂直刚度
        self.d2 = torch.tensor(0.0, device=self.device)
        self.dz = torch.tensor(0.0, device=self.device) # 阻尼
        
        # Dahl 模型参数 (用于低速/静止摩擦)
        self.sigma0 = torch.tensor(100000.0, device=self.device) # 刚度
        self.sigma1 = torch.tensor(5000.0, device=self.device)   # 阻尼
        self.frblend_begin = 1.0 # 混合速度下限 (m/s)
        self.frblend_end = 3.0   # 混合速度上限 (m/s)
        
        # 状态变量 (上一时刻的刚毛变形，用于积分)
        self.brx = None
        self.bry = None
        
        # --- 加载 JSON 配置 ---
        if not os.path.exists(json_filename):
            raise FileNotFoundError(f"JSON not found: {json_filename}")
            
        with open(json_filename, 'r') as f:
            d = json.load(f)
        
        # 调用 Create 方法解析参数
        self.Create(d)

    def Create(self, d):
        """
        重写父类/基类的 Create 方法。
        首先调用 super().Create(d) 以处理 ChPart 的通用参数(Name, Mass, Inertia)，
        然后解析 TMsimple 特有的参数。
        """
        # 1. 基类解析 (Name, Mass, Inertia)
        # 注意：ChPart.Create 会查找 "Design" 下的 Mass/Inertia
        super().Create(d)
        
        # 2. 解析几何参数 (Design)
        if "Design" in d:
            des = d["Design"]
            self.unloaded_radius = torch.tensor(des.get("Unloaded Radius [m]", 0.0), device=self.device)
            self.width = torch.tensor(des.get("Width [m]", 0.0), device=self.device)
            self.rim_radius = torch.tensor(des.get("Rim Radius [m]", 0.0), device=self.device)
        
        # 3. 摩擦与滚阻
        if "Coefficient of Friction" in d:
            self.mu_0 = torch.tensor(d["Coefficient of Friction"], device=self.device)
        
        self.rolling_resistance = torch.tensor(d.get("Rolling Resistance Coefficient", 0.015), device=self.device)

        # 4. 动力学参数 (Parameters)
        if "Parameters" in d:
            p = d["Parameters"]
            # Vertical
            vert = p["Vertical"]
            self.pn = torch.tensor(vert["Nominal Vertical Force [N]"], device=self.device)
            self.dz = torch.tensor(vert["Vertical Tire Damping [Ns/m]"], device=self.device)
            
            # Stiffness
            if "Vertical Tire Stiffness [N/m]" in vert:
                self.d1 = torch.tensor(vert["Vertical Tire Stiffness [N/m]"], device=self.device)
            elif "Tire Spring Curve Data" in vert:
                data = torch.tensor(vert["Tire Spring Curve Data"], device=self.device)
                if data.shape[0] > 1:
                    # 简单线性拟合
                    self.d1 = (data[-1, 1] - data[0, 1]) / (data[-1, 0] - data[0, 0] + 1e-9)
            
            # Longitudinal & Lateral Coefficients
            long = p["Longitudinal"]
            lat = p["Lateral"]
            self._set_coeffs_from_params(long, lat)
            
        elif "Load Index" in d:
            # Load Index 估算逻辑 (对应 Chrono 的自动参数生成)
            self.pn = torch.tensor(8500.0, device=self.device) # HMMWV 近似值
            self.d1 = torch.tensor(400000.0, device=self.device)
            self.dz = torch.tensor(4000.0, device=self.device)
            self._guess_truck_params(self.pn)
            
        print(f"Loaded {self.name} on {self.device}")

    def _set_coeffs_from_params(self, long, lat):
        def get_val(arr): return arr[0] if isinstance(arr, list) else arr
        
        self.par_long = {
            'dfx0_pn': torch.tensor(get_val(long["Initial Slopes dFx/dsx [N]"]), device=self.device),
            'fxm_pn': torch.tensor(get_val(long["Maximum Fx Load [N]"]), device=self.device),
            'fxs_pn': torch.tensor(get_val(long["Sliding Fx Load [N]"]), device=self.device)
        }
        self.par_lat = {
            'dfy0_pn': torch.tensor(get_val(lat["Initial Slopes dFy/dsy [N]"]), device=self.device),
            'fym_pn': torch.tensor(get_val(lat["Maximum Fy Load [N]"]), device=self.device),
            'fys_pn': torch.tensor(get_val(lat["Sliding Fy Load [N]"]), device=self.device)
        }

    def _guess_truck_params(self, pn):
        # 对应 C++ GuessTruck80Par
        self.par_long = {
            'dfx0_pn': 19.9774 * pn, 'fxm_pn': 1.1404 * pn, 'fxs_pn': 0.8448 * pn
        }
        self.par_lat = {
            'dfy0_pn': 16.7895 * pn, 'fym_pn': 1.0107 * pn, 'fys_pn': 0.8486 * pn
        }

    def tm_combined_forces(self, kappa, alpha, fz, muscale):
        """ 
        高速动力学模型 (TMeasy 简化版) 
        """
        dfz = fz / (self.pn + 1e-9)
        
        # 参数提取与载荷修正
        dfx0 = self.par_long['dfx0_pn'] * dfz
        fxm  = self.par_long['fxm_pn'] * dfz
        fxs  = self.par_long['fxs_pn'] * dfz
        
        dfy0 = self.par_lat['dfy0_pn'] * dfz
        fym  = self.par_lat['fym_pn'] * dfz
        fys  = self.par_lat['fys_pn'] * dfz
        
        # 综合滑移
        s = torch.sqrt(kappa**2 + alpha**2) + 1e-9
        
        # 极限包络
        fmax = torch.sqrt((fxm * kappa/s)**2 + (fym * alpha/s)**2)
        dF0  = torch.sqrt((dfx0 * kappa/s)**2 + (dfy0 * alpha/s)**2)
        fs   = torch.sqrt((fxs * kappa/s)**2 + (fys * alpha/s)**2)
        
        # 临界滑移
        s_crit = fmax / (dF0 + 1e-9)
        
        # 曲线计算 (TMeasy 形状近似)
        B = dF0 / (fmax + 1e-9)
        val = B * s
        curve = torch.tanh(val) # 基础饱和
        
        # 衰减因子 (从峰值 Fmax 衰减到滑动值 Fsliding)
        decay = 1.0 - (1.0 - fs/fmax) * torch.sigmoid(10.0 * (s - s_crit))
        
        Fa = muscale * fmax * curve * decay
        
        Fx = Fa * (kappa / s)
        Fy = Fa * (alpha / s)
        
        return Fx, Fy

    def combined_coulomb_forces(self, fz, muscale, step, vx_batch, vy_batch):
        """ 
        低速 Dahl 摩擦模型 (Static/Parking) 
        """
        if self.brx is None or self.brx.shape != vx_batch.shape:
            self.brx = torch.zeros_like(vx_batch)
            self.bry = torch.zeros_like(vy_batch)
            
        fc = fz * muscale
        
        # Dahl 微分方程
        brx_dot = vx_batch - self.sigma0 * self.brx * torch.abs(vx_batch) / (fc + 1e-9)
        Fx = -(self.sigma0 * self.brx + self.sigma1 * brx_dot)
        
        bry_dot = vy_batch - self.sigma0 * self.bry * torch.abs(vy_batch) / (fc + 1e-9)
        Fy = -(self.sigma0 * self.bry + self.sigma1 * bry_dot)
        
        # 状态更新
        self.brx = self.brx + brx_dot * step
        self.bry = self.bry + bry_dot * step
        
        # 摩擦圆限制
        f_mag = torch.sqrt(Fx**2 + Fy**2)
        scale = torch.where(f_mag > fc, fc / (f_mag + 1e-9), torch.ones_like(f_mag))
        
        return Fx * scale, Fy * scale

    def forward(self, kappa, alpha, fz, omega, mu_current=None, vx=None, vy=None, step=1e-3):
        """
        :param vx: 纵向速度 (m/s)，必须提供以支持低速混合
        :param step: 仿真步长 (s)，用于 Dahl 模型积分
        """
        # 输入转 Tensor
        kappa = torch.as_tensor(kappa, device=self.device, dtype=torch.float32)
        alpha = torch.as_tensor(alpha, device=self.device, dtype=torch.float32)
        fz    = torch.as_tensor(fz,    device=self.device, dtype=torch.float32)
        omega = torch.as_tensor(omega, device=self.device, dtype=torch.float32)
        
        if mu_current is None: mu_scale = 1.0
        else: mu_scale = mu_current / self.mu_0
        
        # 必须提供 vx 以判断静止/低速状态
        if vx is None:
            # 默认回退逻辑 (不推荐用于精密测试)
            vx = omega * self.unloaded_radius
        else:
            vx = torch.as_tensor(vx, device=self.device, dtype=torch.float32)
            
        if vy is None:
            vy = vx * torch.tan(alpha)
        else:
            vy = torch.as_tensor(vy, device=self.device, dtype=torch.float32)

        # 1. 计算高速力 (TMeasy)
        Fx_dyn, Fy_dyn = self.tm_combined_forces(kappa, alpha, fz, mu_scale)
        
        # 2. 计算低速力 (Coulomb/Dahl)
        # 滑移速度 vsx = vx - omega * Reff
        # 简化 Reff ≈ Unloaded Radius
        vsx = vx - omega * self.unloaded_radius
        vsy = vy
        Fx_static, Fy_static = self.combined_coulomb_forces(fz, mu_scale, step, vsx, vsy)
        
        # 3. 混合 (Blend)
        abs_vx = torch.abs(vx)
        # 使用 torch.clamp
        blend = (abs_vx - self.frblend_begin) / (self.frblend_end - self.frblend_begin)
        blend = torch.clamp(blend, min=0.0, max=1.0)
        
        Fx = (1.0 - blend) * Fx_static + blend * Fx_dyn
        Fy = (1.0 - blend) * Fy_static + blend * Fy_dyn
        
        # 4. 力矩
        My = -self.rolling_resistance * fz * self.unloaded_radius * torch.tanh(omega)
        Mz = torch.zeros_like(Fx)
        
        # 5. 接触判断
        mask = fz > 0
        zero = torch.zeros_like(Fx)
        
        return {
            'fx': torch.where(mask, Fx, zero),
            'fy': torch.where(mask, Fy, zero),
            'mz': torch.where(mask, Mz, zero),
            'my': torch.where(mask, My, zero),
            'fz': fz
        }

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tire = TMsimpleTire("HMMWV_TMsimpleTire.json", device=device)
    results = tire.forward(
    kappa=torch.tensor([0.1], device=device), 
    alpha=torch.tensor([0.02], device=device), 
    fz=torch.tensor([8500.0], device=device), 
    omega=torch.tensor([10.0], device=device)
    )

    # 2. 通过键名获取具体的力
    fx = results['fx']
    fy = results['fy']
    mz = results['mz']  # 如果需要回正力矩

    print(f"Fx: {fx.item()}, Fy: {fy.item()}")