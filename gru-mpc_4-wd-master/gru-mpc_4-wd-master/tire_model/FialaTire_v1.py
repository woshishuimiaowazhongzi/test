import torch
import numpy as np
from typing import Dict, Tuple, Optional
import json
import os
from ChForceElementTire import ChForceElementTire

class FialaTire(ChForceElementTire):
    """
    基于 Project Chrono FialaTire 的 PyTorch 实现。
    不继承 nn.Module，作为纯物理计算类使用。
    """
    def __init__(self, 
                 config_file: str, 
                 name: str = "FialaTire", 
                 device: Optional[torch.device] = None):

        # 1. 初始化父类（解决原代码重复__init__问题）
        self._device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(name, self._device)
        
        # 2. 加载并解析配置文件
        self.config = self._load_config(config_file)
        self._parse_parameters()
        # 3. 标记初始化完成
        self.is_initialized = True
    def _load_config(self, path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, 'r') as f:
            return json.load(f)

    def _parse_parameters(self):
        # 基础质量惯性属性
        self.mass = torch.tensor(self.config["Mass"], dtype=torch.float32, device=self._device)
        self.inertia = torch.tensor(self.config["Inertia"], dtype=torch.float32, device=self._device)

        # 默认摩擦系数
        self.mu_0 = self.config.get("Coefficient of Friction", 0.8)

        # Fiala 参数
        fp = self.config["Fiala Parameters"]
        self.unloaded_radius = fp["Unloaded Radius"]
        self.width = fp["Width"]
        self.normal_stiffness = fp["Vertical Stiffness"]
        self.normal_damping = fp["Vertical Damping"]
        self.rolling_resistance = fp["Rolling Resistance"]
        self.c_slip = fp["CSLIP"]
        self.c_alpha = fp["CALPHA"]
        self.u_min = fp["UMIN"]
        self.u_max = fp["UMAX"]

        # 可选垂直曲线数据
        self.has_vert_table = False
        self.vert_map_x = None
        self.vert_map_y = None
        self.max_depth = 0.0
        self.max_val = 0.0
        self.slope = 0.0

        if "Vertical Curve Data" in fp:
            data = torch.tensor(fp["Vertical Curve Data"], dtype=torch.float32, device=self._device)
            self.vert_map_x = data[:, 0]
            self.vert_map_y = data[:, 1]
            self.max_val = self.vert_map_y[-1].item()
            dx = self.vert_map_x[-1] - self.vert_map_x[-2]
            dy = self.vert_map_y[-1] - self.vert_map_y[-2]
            self.slope = (dy / dx).item()
            self.has_vert_table = True

        # 初始化状态变量 (用于记录，非必须)
        self.reset_states()

    def reset_states(self):
        self.states = {
            "kappa": torch.zeros(1, device=self._device),
            "alpha": torch.zeros(1, device=self._device),
            "abs_vx": torch.zeros(1, device=self._device),
            "vsx": torch.zeros(1, device=self._device),
            "vsy": torch.zeros(1, device=self._device),
            "omega": torch.zeros(1, device=self._device),
            "abs_vt": torch.zeros(1, device=self._device),
        }

    @staticmethod
    def ch_sine_step_torch(abs_vx, vx_min=0.125, vx_max=0.5, low=0.0, high=1.0):
        """
        PyTorch版ChFunctionSineStep（正弦平滑阶跃函数），兼容自动求导和批量张量
        """
        # 1. 确保输入为Tensor
        if not torch.is_tensor(abs_vx):
            abs_vx = torch.tensor(abs_vx, dtype=torch.float32, device=device)

        # 2. 转换阈值为同设备/类型的Tensor
        vx_min = torch.tensor(vx_min, dtype=abs_vx.dtype, device=abs_vx.device)
        vx_max = torch.tensor(vx_max, dtype=abs_vx.dtype, device=abs_vx.device)
        low = torch.tensor(low, dtype=abs_vx.dtype, device=abs_vx.device)
        high = torch.tensor(high, dtype=abs_vx.dtype, device=abs_vx.device)

        # 3. 初始化输出张量
        out = torch.zeros_like(abs_vx)

        # 4. 分段逻辑
        mask_below = abs_vx <= vx_min
        mask_above = abs_vx >= vx_max
        mask_mid = ~mask_below & ~mask_above

        # 5. 赋值逻辑
        out[mask_below] = low
        out[mask_above] = high

        if torch.any(mask_mid):
            x_norm = (abs_vx[mask_mid] - vx_min) / (vx_max - vx_min)
            out[mask_mid] = 0.5 * (1 - torch.cos(torch.pi * x_norm)) * (high - low) + low

        return out

    def forward(self, kappa, alpha, fz, omega=0.0, mu_road=0.8, vx=None):
        """
        前向计算 Fiala 轮胎力。
        """
        """
        前向计算 Fiala 轮胎力。
        
        Args:
            kappa (Tensor): 纵向滑移率 (Longitudinal slip)
            alpha (Tensor): 侧偏角 (Side slip angle) [rad]
            fz (Tensor): 垂向载荷 (Normal force) [N]
            omega (Tensor): 车轮角速度 [rad/s] (用于计算滚动阻力)
            mu_road (float/Tensor): 当前路面摩擦系数
            
        Returns:
            fx (Tensor): 纵向力 [N]
            fy (Tensor): 侧向力 [N]
            fz (Tensor): 垂向力 [N]
            mz (Tensor): 回正力矩 [Nm]
            my (Tensor): 滚动阻力矩 [Nm]
            mx (Tensor): 侧向力矩 [Nm]
        """
        # 物理限制：Fz 必须非负
        fz = torch.clamp(fz, min=0.0)

        # 1. 计算综合滑移量 (Combined Slip)
        tan_alpha = torch.tan(alpha)
        SsA = torch.sqrt(kappa.pow(2) + tan_alpha.pow(2))
        SsA = torch.clamp(SsA, max=1.0)

        # 2. 计算综合摩擦系数 U
        U = self.u_max - (self.u_max - self.u_min) * SsA
        # 根据当前路面摩擦系数调整 U
        U = U * (mu_road / self.mu_0)

        # 防止 U * fz 为 0 导致后续除法 NaN
        ufz = U * fz
        ufz_safe = torch.where(ufz < 1e-6, torch.ones_like(ufz) * 1e-6, ufz)

        # 3. 计算临界值 (Critical Values)
        s_critical = torch.abs(ufz / (2.0 * self.c_slip))
        alpha_critical = torch.atan(3.0 * ufz / self.c_alpha)

        # ==========================
        # 4. 纵向力 Fx 计算
        # ==========================

        # Case A: 弹性区 (Elastic Region)
        fx_elastic = self.c_slip * kappa

        # Case B: 滑动区 (Sliding Region)
        kappa_safe = torch.where(torch.abs(kappa) < 1e-6, torch.ones_like(kappa) * 1e-6, kappa)
        fx2_term = (ufz).pow(2) / (4.0 * torch.abs(kappa_safe) * self.c_slip)
        fx_sliding = torch.sign(kappa) * (ufz - fx2_term)

        # 组合 Fx
        fx = torch.where(torch.abs(kappa) < s_critical, fx_elastic, fx_sliding)

        # ==========================
        # 5. 侧向力 Fy 和 回正力矩 Mz 计算
        # ==========================

        abs_alpha = torch.abs(alpha)
        abs_tan_alpha = torch.abs(tan_alpha)
        sign_alpha = torch.sign(alpha)

        # Case A: 弹性/过渡区
        H = 1.0 - (self.c_alpha * abs_tan_alpha) / (3.0 * ufz_safe)

        fy_elastic = -ufz * (1.0 - H.pow(3)) * sign_alpha
        mz_elastic = ufz * self.width * (1.0 - H) * H.pow(3) * sign_alpha

        # Case B: 完全滑动区
        fy_sliding = -ufz * sign_alpha
        mz_sliding = torch.zeros_like(mz_elastic)  # Mz 归零

        # 组合 Fy, Mz
        fy = torch.where(abs_alpha <= alpha_critical, fy_elastic, fy_sliding)
        mz = torch.where(abs_alpha <= alpha_critical, mz_elastic, mz_sliding)

        # ==========================
        # 6. 滚动阻力矩 My
        # // Smoothing factor dependend on m_state.abs_vx, allows soft switching of My
        # double myStartUp = ChFunctionSineStep::Eval(m_states.abs_vx, vx_min, 0.0, vx_max, 1.0);
        # 该函数是「软阶跃 / 平滑阶跃」（区别于硬阶跃 sign 函数），在 vx_min ~ vx_max 区间内通过正弦曲线实现从 0.0 到 1.0 的平滑过渡，避免硬阶跃的数值突变（防止仿真震荡）。
        # Ref: ChFialaTire.cpp:138 -> My = -myStartUp * m_rolling_resistance * m_data.normal_force * ChSignum(m_states.omega);
        # ==========================

        if vx is not None:
            abs_vx = torch.abs(vx)
            vx_min = 0.125
            vx_max = 0.5
            my_startup = self.ch_sine_step_torch(abs_vx, vx_min, vx_max)
            my = -my_startup * self.rolling_resistance * fz * self.unloaded_radius * torch.sign(omega)
        else:
            my = -self.rolling_resistance * fz * self.unloaded_radius * torch.sign(omega)

        mx = torch.zeros_like(my)

        forces = {
            "fx": fx,
            "fy": fy,
            "fz": fz,
            "mx": mx,
            "my": my,
            "mz": mz
        }

        return forces

# =============================================================================
# 测试用例
# =============================================================================
if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用计算设备: {device}")
    # 2. 轮胎初始化（指定配置文件+设备，确保与输入张量一致）
    config_path = "./HMMWV_FialaTire.json"

    tire = FialaTire(config_file=config_path, device=device)
    kappa_batch = torch.tensor([0.01, 0.05, 0.2, -0.1, 0.0], dtype=torch.float32, device=device) # 滑移率
    alpha_batch = torch.tensor([0.0,  0.02, 0.05, 0.1, 0.2], dtype=torch.float32, device=device) # 侧偏角 [rad]
    fz_batch    = torch.tensor([4000.0, 4000.0, 4000.0, 5000.0, 0], dtype=torch.float32, device=device) # 载荷 (最后一个离地)
    omega_batch = torch.tensor([40.0, 10.0, 60.0, 20.0, 0.0], dtype=torch.float32, device=device)
        
    results = tire.forward(kappa_batch, alpha_batch, fz_batch, omega_batch)
        
    # 打印输出（从results字典取张量，detach解除梯度关联，cpu兼容CUDA，转NumPy）
    print("--- Fiala Tire Force Output ---")
    print(f"Fx (Longitudinal): {results['fx'].detach().cpu().numpy()}")
    print(f"Fy (Lateral):      {results['fy'].detach().cpu().numpy()}")
    print(f"Mz (Aligning):     {results['mz'].detach().cpu().numpy()}")
    print(f"My (Rolling):      {results['my'].detach().cpu().numpy()}")
    print(f"Fz (Vertical):     {results['fz'].detach().cpu().numpy()}")
    print(f"Mx (Torsional):    {results['mx'].detach().cpu().numpy()}")
  
  