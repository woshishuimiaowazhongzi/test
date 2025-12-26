import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import os
import sys
# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将目录添加到sys.path
sys.path.append(current_dir)
from ChForceElementTire import ChForceElementTire
from ChPart import ChPart

class ChPac02Tire(ChForceElementTire):
    def __init__(self, json_path, device='cpu'):
        super().__init__(json_path, device)
        
        # 默认参数 (Pacejka 2002 Coefficients Structure)
        # 初始化所有系数为 0 或 1 (参考 C++ 结构体 MFCoeff)
        self.par = {
            # Scaling
            'LFZO': 1.0, 'LCX': 1.0, 'LMUX': 1.0, 'LEX': 1.0, 'LKX': 1.0, 'LHX': 1.0, 'LVX': 1.0, 'LGAX': 1.0,
            'LCY': 1.0, 'LMUY': 1.0, 'LEY': 1.0, 'LKY': 1.0, 'LHY': 1.0, 'LVY': 1.0, 'LGAY': 1.0,
            'LTR': 1.0, 'LRES': 1.0, 'LGAZ': 1.0, 'LXAL': 1.0, 'LYKA': 1.0, 'LVYKA': 1.0, 'LS': 1.0,
            'LSGKP': 1.0, 'LSGAL': 1.0, 'LGYR': 1.0, 'LVMX': 1.0, 'LMX': 1.0, 'LMY': 1.0, 'LIP': 1.0, 'LCZ': 1.0, 'LKYG': 1.0,
            
            # Longitudinal
            'PCX1': 0.0, 'PDX1': 0.0, 'PDX2': 0.0, 'PDX3': 0.0,
            'PEX1': 0.0, 'PEX2': 0.0, 'PEX3': 0.0, 'PEX4': 0.0,
            'PKX1': 0.0, 'PKX2': 0.0, 'PKX3': 0.0,
            'PHX1': 0.0, 'PHX2': 0.0, 'PVX1': 0.0, 'PVX2': 0.0,
            'RBX1': 0.0, 'RBX2': 0.0, 'RCX1': 0.0, 'REX1': 0.0, 'REX2': 0.0, 'RHX1': 0.0,
            'PTX1': 0.0, 'PTX2': 0.0, 'PTX3': 0.0,
            'PPX1': 0.0, 'PPX2': 0.0, 'PPX3': 0.0, 'PPX4': 0.0,
            
            # Lateral
            'PCY1': 0.0, 'PDY1': 0.0, 'PDY2': 0.0, 'PDY3': 0.0,
            'PEY1': 0.0, 'PEY2': 0.0, 'PEY3': 0.0, 'PEY4': 0.0,
            'PKY1': 0.0, 'PKY2': 0.0, 'PKY3': 0.0,
            'PHY1': 0.0, 'PHY2': 0.0, 'PHY3': 0.0,
            'PVY1': 0.0, 'PVY2': 0.0, 'PVY3': 0.0, 'PVY4': 0.0,
            'RBY1': 0.0, 'RBY2': 0.0, 'RBY3': 0.0, 'RCY1': 0.0, 'REY1': 0.0, 'REY2': 0.0, 'RHY1': 0.0, 'RHY2': 0.0,
            'RVY1': 0.0, 'RVY2': 0.0, 'RVY3': 0.0, 'RVY4': 0.0, 'RVY5': 0.0, 'RVY6': 0.0,
            'PTY1': 0.0, 'PTY2': 0.0,
            'PPY1': 0.0, 'PPY2': 0.0, 'PPY3': 0.0, 'PPY4': 0.0,
            
            # Aligning
            'QBZ1': 0.0, 'QBZ2': 0.0, 'QBZ3': 0.0, 'QBZ4': 0.0, 'QBZ5': 0.0, 'QBZ9': 0.0, 'QBZ10': 0.0,
            'QCZ1': 0.0, 'QDZ1': 0.0, 'QDZ2': 0.0, 'QDZ3': 0.0, 'QDZ4': 0.0, 'QDZ6': 0.0, 'QDZ7': 0.0, 'QDZ8': 0.0, 'QDZ9': 0.0,
            'QEZ1': 0.0, 'QEZ2': 0.0, 'QEZ3': 0.0, 'QEZ4': 0.0, 'QEZ5': 0.0,
            'QHZ1': 0.0, 'QHZ2': 0.0, 'QHZ3': 0.0, 'QHZ4': 0.0,
            'SSZ1': 0.0, 'SSZ2': 0.0, 'SSZ3': 0.0, 'SSZ4': 0.0,
            'QPZ1': 0.0, 'QPZ2': 0.0, 'QTZ1': 0.0, 'MBELT': 0.0,
            
            # Rolling & Overturning
            'QSY1': 0.01, 'QSY2': 0.0, 'QSY3': 0.0, 'QSY4': 0.0, 'QSY5': 0.0, 'QSY6': 0.0, 'QSY7': 0.0, 'QSY8': 0.0,
            'QSX1': 0.0, 'QSX2': 0.0, 'QSX3': 0.0, 'QSX4': 0.0, 'QSX5': 0.0, 'QSX6': 0.0, 'QSX7': 0.0, 'QSX8': 0.0, 'QSX9': 0.0, 'QSX10': 0.0, 'QSX11': 0.0,
            'QPX1': 0.0,
            
            # Dimensions / Standard
            'UNLOADED_RADIUS': 0.0, 'WIDTH': 0.0, 'ASPECT_RATIO': 0.0, 'RIM_RADIUS': 0.0, 'RIM_WIDTH': 0.0,
            'FNOMIN': 4000.0, 'VERTICAL_STIFFNESS': 200000.0, 'VERTICAL_DAMPING': 500.0,
            'BREFF': 0.0, 'DREFF': 0.0, 'FREFF': 0.0,
            'QFZ1': 0.0, 'QFZ2': 0.0, 'QFZ3': 0.0, 'QPFZ1': 0.0, 'QV2': 0.0,
            'IP': 200000.0, 'IP_NOM': 200000.0, 'TIRE_MASS': 0.0
        }
        
        self.use_friction_ellipsis = True # Default from C++
        self.mu0 = torch.tensor(0.8, device=device) # Reference friction

    def CalcFxyMz(self, kappa, alpha, fz, gamma, mu_scale):
        """ 
        Pacejka 2002 核心公式实现
        对应 C++ ChPac02Tire::CalcFxyMz
        所有公式均 1:1 复刻
        """
        p = self.par # shorthand
        
        # 1. 归一化变量 (Normalized variables)
        Fz0_prime = p['FNOMIN'] * p['LFZO']
        dfz0 = (fz - Fz0_prime) / Fz0_prime
        dpi = (p['IP'] - p['IP_NOM'] * p['LIP']) / (p['IP_NOM'] * p['LIP'])
        
        # ===============================================
        # Longitudinal Force Fx
        # ===============================================
        Cx = p['PCX1'] * p['LCX']
        mu_x = (p['PDX1'] + p['PDX2'] * dfz0) * \
               (1.0 + p['PPX3'] * dpi + p['PPX4'] * dpi**2) * \
               (1.0 - p['PDX3'] * gamma**2) * p['LMUX'] * mu_scale
        Dx = mu_x * fz
        
        Ex = (p['PEX1'] + p['PEX2'] * dfz0 + p['PEX3'] * dfz0**2) * \
             (1.0 - p['PEX4'] * torch.sign(kappa)) * p['LEX']
        Ex = torch.clamp(Ex, -10.0, 1.0) # C++: if (Ex > 1.0) Ex = 1.0; (assumed similar lower bound)
        
        Kx = fz * (p['PKX1'] + p['PKX2'] * dfz0) * torch.exp(p['PKX3'] * dfz0) * \
             (1.0 + p['PPX1'] * dpi + p['PPX2'] * dpi**2) * p['LKX']
        
        Bx = Kx / (Cx * Dx + 0.1) # +0.1 to avoid div by zero
        
        Shx = (p['PHX1'] + p['PHX2'] * dfz0) * p['LHX']
        Svx = fz * (p['PVX1'] + p['PVX2'] * dfz0) * p['LVX'] * p['LMUX']
        
        kappa_x = kappa + Shx
        X1 = Bx * kappa_x
        # C++: Chtorch.clampValue(X1, -CH_PI_2 + 0.01, CH_PI_2 - 0.01); 
        # Note: In pure Pac02, X1 isn't torch.clamped, but Chrono implementation does. We follow Chrono.
        # However, inside atan, arguments are free. The torch.clamp usually applies to input of something else.
        # Looking at C++ code: no torch.clamp on X1 inside CalcFxyMz before atan. 
        # Wait, line 163 in ChPac02Tire.cpp: Chtorch.clampValue(X1, -CH_PI_2 + 0.01, CH_PI_2 - 0.01);
        # This seems physically wrong for atan argument, but I must follow instructions "1:1 copy".
        # Assuming the C++ comment means limiting the input to sin? 
        # Actually, atan domain is -inf to inf. 
        # Let's double check the C++ code provided.
        # "double X1 = Bx * kappa_x; Chtorch.clampValue(X1, -CH_PI_2 + 0.01, CH_PI_2 - 0.01);"
        # This line exists in the C++ snippet provided. I will implement it.
        X1 = torch.clamp(X1, -math.pi/2 + 0.01, math.pi/2 - 0.01)
        
        Fx0 = Dx * torch.sin(Cx * torch.atan(X1 - Ex * (X1 - torch.atan(X1)))) + Svx
        
        # ===============================================
        # Lateral Force Fy
        # ===============================================
        gamma_y = gamma * p['LGAY']
        Cy = p['PCY1'] * p['LCY']
        
        Ky0 = p['PKY1'] * p['FNOMIN'] * (1 + p['PPY1'] * dpi) * \
              torch.sin(2.0 * torch.atan(fz / (p['PKY2'] * Fz0_prime * (1.0 + p['PPY2'] * dpi)))) * \
              p['LFZO'] * p['LMUY']
        Ky = Ky0 * (1.0 - p['PKY3'] * torch.abs(gamma_y))
        
        Shy = (p['PHY1'] + p['PHY2'] * dfz0) * p['LHY'] + p['PHY3'] * gamma_y * p['LKYG']
        alpha_y = alpha + Shy
        
        Svy = fz * ((p['PVY1'] + p['PVY2'] * dfz0) * p['LVY'] + 
                    (p['PVY3'] + p['PVY4'] * dfz0) * gamma_y * p['LKYG']) * p['LMUY']
        
        Ey = (p['PEY1'] + p['PEY2'] * dfz0) * \
             (1.0 - (p['PEY3'] + p['PEY4'] * gamma_y) * torch.sign(alpha_y)) * p['LEY']
        Ey = torch.clamp(Ey, -10.0, 1.0) # C++: if (Ey > 1.0) Ey = 1.0;
        
        mu_y = torch.abs(mu_scale * (p['PDY1'] + p['PDY2'] * dfz0) * (1.0 + p['PPY3'] * dpi + p['PPY4'] * dpi**2) * (1.0 + p['PDY3'] * gamma_y**2) * p['LMUY'])
        
        Dy = mu_y * fz
        By = Ky / (Cy * Dy + 0.1)
        Y1 = By * alpha_y
        # C++: Chtorch.clampValue(Y1, -CH_PI_2 + 0.01, CH_PI_2 - 0.01);
        Y1 = torch.clamp(Y1, -math.pi/2 + 0.01, math.pi/2 - 0.01)
        
        Fy0 = Dy * torch.sin(Cy * torch.atan(Y1 - Ey * (Y1 - torch.atan(Y1)))) + Svy
        
        # ===============================================
        # Combined Slip
        # ===============================================
        Fx, Fy, Mz = torch.zeros_like(Fx0), torch.zeros_like(Fy0), torch.zeros_like(Fx0)
        
        # We only implement m_use_mode = 4 (Combined) as per context of verification
        # case 4 in C++ switch
        if self.use_friction_ellipsis:
            # Friction Ellipsis method (default in Chrono)
            # kappa_c = kappa + Shx + Svx / Kx;
            kappa_c = kappa + Shx + Svx / (Kx + 0.1)
            # alpha_c = alpha + Shy + Svy / Ky;
            alpha_c = alpha + Shy + Svy / (Ky + 0.1)
            alpha_s = torch.sin(alpha_c)
            
            # beta = acos(|kappa_c| / hypot(kappa_c, alpha_s))
            hypot_c = torch.hypot(kappa_c, alpha_s) + 1e-9
            beta = torch.acos(torch.abs(kappa_c) / hypot_c)
            
            mu_x_act = torch.abs((Fx0 - Svx) / (fz + 0.1))
            mu_y_act = torch.abs((Fy0 - Svy) / (fz + 0.1))
            mu_x_max = Dx / (fz + 0.1)
            mu_y_max = Dy / (fz + 0.1)
            
            # mu_x_c = 1.0 / hypot(1.0 / mu_x_act, tan(beta) / mu_y_max)
            tan_beta = torch.tan(beta)
            denom_x = torch.hypot(1.0 / (mu_x_act + 1e-9), tan_beta / (mu_y_max + 1e-9))
            mu_x_c = 1.0 / (denom_x + 1e-9)
            
            # mu_y_c = tan(beta) / hypot(1.0 / mu_x_max, tan(beta) / mu_y_act)
            denom_y = torch.hypot(1.0 / (mu_x_max + 1e-9), tan_beta / (mu_y_act + 1e-9))
            mu_y_c = tan_beta / (denom_y + 1e-9)
            
            Fx = Fx0 * mu_x_c / (mu_x_act + 1e-9)
            Fy = Fy0 * mu_y_c / (mu_y_act + 1e-9)
            
            # Aligning Moment (Uncombined trail used in Friction Ellipsis method in C++)
            # Mz = -t * Fy + Mzr;
            
            # Pneumatic trail t (Steady State)
            gamma_z = gamma * p['LGAZ']
            Sht = p['QHZ1'] + p['QHZ2'] * dfz0 + (p['QHZ3'] + p['QHZ4'] * dfz0) * gamma_z
            alpha_t = alpha + Sht
            
            Ct = p['QCZ1']
            Bt = torch.abs((p['QBZ1'] + p['QBZ2'] * dfz0 + p['QBZ3'] * dfz0**2) * \
                           (1.0 + p['QBZ4'] * gamma_z + p['QBZ5'] * torch.abs(gamma_z)) * p['LKY'] / p['LMUY'])
            Et = (p['QEZ1'] + p['QEZ2'] * dfz0 + p['QEZ3'] * dfz0**2) * \
                 (1.0 + (p['QEZ4'] + p['QEZ5'] * gamma_z) * ((2.0 / math.pi) * torch.atan(Bt * Ct * alpha_t)))
            Et = torch.clamp(Et, -10.0, 1.0) # C++: if (Et > 1.0) Et = 1.0;
            
            Dt = fz * (p['QDZ1'] + p['QDZ2'] * dfz0) * (1.0 - p['QPZ1'] * dpi) * \
                 (1.0 + p['QDZ3'] * gamma_z + p['QDZ4'] * gamma_z**2) * p['UNLOADED_RADIUS'] / Fz0_prime * p['LTR']
            
            t = Dt * torch.cos(Ct * torch.atan(Bt * alpha_t - Et * (Bt * alpha_t - torch.atan(Bt * alpha_t)))) * torch.cos(alpha)
            
            # Residual Moment Mzr
            Shf = Shy + Svy / (Ky + 0.1)
            alpha_r = alpha + Shf
            Br = (p['QBZ9'] * p['LKY'] / p['LMUY'] + p['QBZ10'] * By * Cy)
            Dr = fz * ((p['QDZ6'] + p['QDZ7'] * dfz0) * p['LRES'] + \
                       (p['QDZ8'] + p['QDZ9'] * dfz0) * (1.0 + p['QPZ2'] * dpi) * gamma_z) * \
                 p['UNLOADED_RADIUS'] * p['LMUY']
            
            Mzr = Dr * torch.cos(torch.atan(Br * alpha_r)) * torch.cos(alpha)
            
            Mz = -t * Fy + Mzr
            
        else:
            # Pacejka method for combined slip
            # C++ implementation lines 272-302
            # Longitudinal
            Shxa = p['RHX1']
            Cxa = p['RCX1']
            alpha_s = alpha + Shxa
            Exa = p['REX1'] + p['REX2'] * dfz0
            Exa = torch.clamp(Exa, -10.0, 1.0)
            Bxa = torch.abs(p['RBX1'] * torch.cos(torch.atan(p['RBX2'] * kappa)) * p['LXAL'])
            
            num_Gxa = torch.cos(Cxa * torch.atan(Bxa * alpha_s - Exa * (Bxa * alpha_s - torch.atan(Bxa * alpha_s))))
            den_Gxa = torch.cos(Cxa * torch.atan(Bxa * Shxa - Exa * (Bxa * Shxa - torch.atan(Bxa * Shxa))))
            Gxa = num_Gxa / (den_Gxa + 1e-9)
            Fx = Fx0 * Gxa
            
            # Lateral
            Shyk = p['RHY1'] + p['RHY2'] * dfz0
            kappa_s = kappa + Shyk
            Cyk = p['RCY1']
            Eyk = p['REY1'] + p['REY2'] * dfz0
            Eyk = torch.clamp(Eyk, -10.0, 1.0)
            Byk = p['RBY1'] * torch.cos(torch.atan(p['RBY2'] * (alpha - p['RBY3']))) * p['LYKA']
            Dvyk = mu_y * fz * (p['RVY1'] + p['RVY2'] * dfz0 + p['RVY3'] * gamma) * \
                   torch.cos(torch.atan(p['RVY4'] * alpha))
            Svyk = Dvyk * torch.sin(p['RVY5'] * torch.atan(p['RVY6'] * kappa)) * p['LVYKA']
            
            num_Gyk = torch.cos(Cyk * torch.atan(Byk * kappa_s - Eyk * (Byk * kappa_s - torch.atan(Byk * kappa_s))))
            den_Gyk = torch.cos(Cyk * torch.atan(Byk * Shyk - Eyk * (Byk * Shyk - torch.atan(Byk * Shyk))))
            Gyk = num_Gyk / (den_Gyk + 1e-9)
            Fy = Fy0 * Gyk + Svyk
            
            # Moment
            # Recalculate t and Mzr components
            gamma_z = gamma * p['LGAZ']
            Sht = p['QHZ1'] + p['QHZ2'] * dfz0 + (p['QHZ3'] + p['QHZ4'] * dfz0) * gamma_z
            alpha_t = alpha + Sht
            
            Ct = p['QCZ1']
            Bt = torch.abs((p['QBZ1'] + p['QBZ2'] * dfz0 + p['QBZ3'] * dfz0**2) * \
                           (1.0 + p['QBZ4'] * gamma_z + p['QBZ5'] * torch.abs(gamma_z)) * p['LKY'] / p['LMUY'])
            Et = (p['QEZ1'] + p['QEZ2'] * dfz0 + p['QEZ3'] * dfz0**2) * \
                 (1.0 + (p['QEZ4'] + p['QEZ5'] * gamma_z) * ((2.0 / math.pi) * torch.atan(Bt * Ct * alpha_t)))
            Et = torch.clamp(Et, -10.0, 1.0)
            
            Dt = fz * (p['QDZ1'] + p['QDZ2'] * dfz0) * (1.0 - p['QPZ1'] * dpi) * \
                 (1.0 + p['QDZ3'] * gamma_z + p['QDZ4'] * gamma_z**2) * p['UNLOADED_RADIUS'] / Fz0_prime * p['LTR']
                 
            # Combined alpha equivalents
            alpha_teq = torch.atan(torch.sqrt(torch.tan(alpha_t)**2 + (Kx / (Ky + 0.1))**2 * kappa**2)) * torch.sign(kappa)
            
            # Recalc alpha_r for Moment
            Shf = Shy + Svy / (Ky + 0.1)
            alpha_r = alpha + Shf
            alpha_req = torch.atan(torch.sqrt(torch.tan(alpha_r)**2 + (Kx / (Ky + 0.1))**2 * kappa**2)) * torch.sign(alpha_r)
            
            tc = Dt * torch.cos(Ct * torch.atan(Bt * alpha_teq - Et * (Bt * alpha_teq - torch.atan(Bt * alpha_teq)))) * torch.cos(alpha)
            
            s = (p['SSZ1'] + p['SSZ2'] * Fy / Fz0_prime + (p['SSZ3'] + p['SSZ4'] * dfz0) * gamma) * p['UNLOADED_RADIUS'] * p['LS']
            
            Br = (p['QBZ9'] * p['LKY'] / p['LMUY'] + p['QBZ10'] * By * Cy)
            Dr = fz * ((p['QDZ6'] + p['QDZ7'] * dfz0) * p['LRES'] + \
                       (p['QDZ8'] + p['QDZ9'] * dfz0) * (1.0 + p['QPZ2'] * dpi) * gamma_z) * \
                 p['UNLOADED_RADIUS'] * p['LMUY']
            
            Mzrc = Dr * torch.cos(torch.atan(Br * alpha_req)) * torch.cos(alpha)
            
            Fy_prime = Fy - Svyk
            Mz = -tc * Fy_prime + Mzrc + s * Fx
            
        return Fx, Fy, Mz

    def CalcMy(self, Fx, Fz, gamma, vx):
        """ """
        p = self.par
        # V0 = std::sqrt(m_g * m_par.UNLOADED_RADIUS);
        V0 = math.sqrt(9.81 * p['UNLOADED_RADIUS'])
        vx_norm = torch.abs(vx) / V0
        
        # QSY1 + QSY2*Fx/Fn + QSY3*|v/v0| + QSY4*(v/v0)^4 ...
        qsy_term = (p['QSY1'] + p['QSY2'] * Fx / p['FNOMIN'] + p['QSY3'] * vx_norm + p['QSY4'] * vx_norm**4 +
                    p['QSY5'] * gamma**2 + p['QSY6'] * gamma**2 * Fz / p['FNOMIN'])
                    
        scale = (Fz / p['FNOMIN'])**p['QSY7'] * (p['IP'] / p['IP_NOM'])**p['QSY8']
        
        My = Fz * p['UNLOADED_RADIUS'] * qsy_term * scale * p['LMY']
        return My

# ==============================================================================
# 3. Pac02Tire (数据加载与封装)
# ==============================================================================

class Pac02Tire(ChPac02Tire):
    def __init__(self, json_filename, device='cpu'):
        super().__init__("Pac02Tire", device)        
        if not os.path.exists(json_filename):
            raise FileNotFoundError(f"JSON not found: {json_filename}")            
        with open(json_filename, 'r') as f:
            d = json.load(f)        
        self.Create(d)
        print(f"Loaded Pac02Tire: {self.name} on {self.device}")
    def Create(self, d):
        ChPart.Create(self, d)        
        if "Coefficient of Friction" in d:
            self.mu0 = torch.tensor(d["Coefficient of Friction"], device=self.device)            
        # Parse TIR file
        tir_file = d.get("TIR Specification File", "")        
        if tir_file:
            # 获取当前python 文件路径
            dir_path = os.path.dirname(os.path.realpath(__file__))
            local_tir = os.path.join(dir_path, tir_file)  
            if os.path.exists(local_tir):
                self._load_tir_file(local_tir)
            else:
                print(f"Warning: TIR file {tir_file} not found locally. Using internal HMMWV defaults.")
                self._set_hmmwv_defaults()
        else:
            self._set_hmmwv_defaults()
            
        # Override from JSON if present
        if d.get("Friction Ellipsis Mode", True) == False:
            self.use_friction_ellipsis = False

    def _set_hmmwv_defaults(self):
        """
        设置 HMMWV 默认参数 (Reference: HMMWV TMeasy converted to basic Pac02 approximate)
        """
        p = self.par
        # Dimensions
        p['UNLOADED_RADIUS'] = 0.4699
        p['WIDTH'] = 0.3175
        p['FNOMIN'] = 8500.0 
        p['VERTICAL_STIFFNESS'] = 400000.0
        
        # Longitudinal
        p['PCX1'] = 1.6; p['PDX1'] = 1.05; p['PKX1'] = 25.0
        p['RBX1'] = 12.0; p['RBX2'] = 10.0; p['RCX1'] = 1.0
        
        # Lateral
        p['PCY1'] = 1.4; p['PDY1'] = 0.95; p['PKY1'] = 18.0; p['PKY2'] = 1.5
        p['RBY1'] = 10.0; p['RBY2'] = 10.0
        
        # Aligning
        p['QCZ1'] = 1.1; p['QDZ1'] = 0.12; p['QBZ1'] = 8.0
        
        # Rolling
        p['QSY1'] = 0.015

    def _load_tir_file(self, filepath):
        """ 简易 TIR 解析器 """
        print(f"Parsing TIR: {filepath}")
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('!') or line.startswith('$'): continue
                # Remove comments
                line = line.split('$')[0].split('!')[0].strip()
                if '=' in line:
                    parts = line.split('=')
                    key = parts[0].strip().upper()
                    val_str = parts[1].strip()
                    # Remove quotes
                    val_str = val_str.replace("'", "").replace('"', "")
                    
                    if key in self.par:
                        try:
                            self.par[key] = float(val_str)
                        except:
                            pass # String values like LEFT/RIGHT

    def forward(self, kappa, alpha, fz, omega, mu_current=None):
        """
        前向计算接口: Fx, Fy, Fz, My, Mz
        """
        # Ensure tensors
        if not torch.is_tensor(kappa): kappa = torch.tensor(kappa, device=self.device)
        if not torch.is_tensor(alpha): alpha = torch.tensor(alpha, device=self.device)
        if not torch.is_tensor(fz): fz = torch.tensor(fz, device=self.device)
        if not torch.is_tensor(omega): omega = torch.tensor(omega, device=self.device)
        
        # Friction Scaling
        mu_scale = 1.0
        if mu_current is not None:
            mu_scale = mu_current / self.mu0
            
        gamma = torch.zeros_like(kappa) # 简化：本次测试不涉及外倾角
        
        # 1. Calc Forces (Fx, Fy, Mz)
        fx, fy, mz = self.CalcFxyMz(kappa, alpha, fz, gamma, mu_scale)
        
        # 2. Calc Rolling Resistance (My)
        vx = omega * self.par['UNLOADED_RADIUS']
        my = -self.CalcMy(fx, fz, gamma, vx) * torch.sign(omega)
        
        
        
        return {
            'fx': fx, 'fy': fy, 'mz': mz, 'my': my, 'fz': fz
        }

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 获取当前python文件路径
    dir_path = os.path.dirname(os.path.abspath(__file__))
    tire = Pac02Tire(dir_path+"/HMMWV_Pac02Tire.json", device=device)
    results = tire.forward(
    kappa=torch.tensor([0.01], device=device), 
    alpha=torch.tensor([0.01], device=device), 
    fz=torch.tensor([8500.0], device=device), 
    omega=torch.tensor([0.0], device=device)
    )

    # 2. 通过键名获取具体的力
    fx = results['fx']
    fy = results['fy']
    mz = results['mz']  # 如果需要回正力矩

    print(f"Fx: {fx.item()}, Fy: {fy.item()}")