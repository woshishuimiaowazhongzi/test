import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import re
from Pac02Tire import Pac02Tire

# 全局设备设置

def preprocess_results(results):
    plot_data = {}
    for key, val in results.items():
        plot_data[key] = val.cpu().numpy().squeeze()
    return plot_data

def plot_multi_load_results(results_list, fz_labels, x_data, x_label, title, save_name, save_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    colors = ['r', 'g', 'b', 'c', 'm']

    if 'slip angle' in x_label.lower():
        # Case 1: Alpha Sweep -> Plot Fy, Mz
        key1, label1 = 'fy', 'Lateral Force (Fy)'
        key2, label2 = 'mz', 'Aligning Moment (Mz)'
    else:
        # Case 2: Kappa Sweep -> Plot Fx, Fy
        key1, label1 = 'fx', 'Longitudinal Force (Fx)'
        key2, label2 = 'fy', 'Lateral Force (Fy)'

    for idx, (res, label) in enumerate(zip(results_list, fz_labels)):
        if key1 in res:
            ax1.plot(x_data, res[key1], color=colors[idx % len(colors)], linewidth=2, label=f"{label}")
    ax1.set_ylabel(f'{label1} [N]', fontsize=12)
    ax1.grid(True, alpha=0.3); ax1.legend()

    for idx, (res, label) in enumerate(zip(results_list, fz_labels)):
        if key2 in res:
            ax2.plot(x_data, res[key2], color=colors[idx % len(colors)], linestyle='--', linewidth=2, label=f"{label}")
    ax2.set_xlabel(x_label, fontsize=12)
    ax2.set_ylabel(f'{label2} [N or Nm]', fontsize=12)
    ax2.grid(True, alpha=0.3); ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, save_name), dpi=150, bbox_inches='tight')
    plt.close()

# ==============================================================================
# 3. 主程序执行逻辑
# ==============================================================================

if __name__ == "__main__":
    # 1. 准备目录
    # 当前python文件路径
    dir_path = os.path.dirname(os.path.abspath(__file__))
    
    save_dir = os.path.join(dir_path, "pac02_optimized_results")
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 准备 TIR 文件内容 (嵌入 HMMWV 参数)
    tir_content = """
    [UNITS]
    LENGTH                   = 'meter'
    FORCE                    = 'newton'
    ANGLE                    = 'radians'
    MASS                     = 'kg'
    TIME                     = 'second'
    [MODEL]
    FITTYP                   = 6
    USE_MODE                 = 4
    LONGVL                   = 16.6
    [DIMENSION]
    UNLOADED_RADIUS          = 0.4699
    WIDTH                    = 0.3175
    ASPECT_RATIO             = 0.75
    RIM_RADIUS               = 0.2095
    RIM_WIDTH                = 0.2
    [VERTICAL]
    VERTICAL_STIFFNESS       = 411121.0
    VERTICAL_DAMPING         = 3900.0
    FNOMIN                   = 8562.0
    [SCALING_COEFFICIENTS]
    LFZO                     = 1.0
    [LONGITUDINAL_COEFFICIENTS]
    PCX1                     = 1.55
    PDX1                     = 1.02
    PKX1                     = 28.0
    RBX1                     = 12.3
    RCX1                     = 1.08
    [LATERAL_COEFFICIENTS]
    PCY1                     = 1.35
    PDY1                     = 0.95
    PKY1                     = 18.5
    PKY2                     = 1.6
    RBY1                     = 10.0
    RCY1                     = 1.05
    [ALIGNING_COEFFICIENTS]
    QCZ1                     = 1.15
    QDZ1                     = 0.11
    QBZ1                     = 9.0
    [ROLLING_COEFFICIENTS]
    QSY1                     = 0.015
    """
    tir_path = "HMMWV_Pac02Tire.tir"
    with open(tir_path, 'w') as f: f.write(tir_content)
    
    # 3. 准备 JSON 文件
    json_path = "HMMWV_Pac02Tire.json"
    json_data = {
        "Name": "HMMWV Pacejka Test",
        "Mass": 37.6,
        "Inertia": [3.84, 6.69, 3.84],
        "TIR Specification File": tir_path,
        "Coefficient of Friction": 0.8
    }
    with open(json_path, 'w') as f: json.dump(json_data, f)
    
    # 4. 初始化模型
    tire = Pac02Tire(json_path, device=device)
    
    # 5. 测试工况 (与 Fiala 保持一致)
    test_loads = [2000.0, 4000.0, 6000.0, 8500.0]
    fz_labels = [f"Fz={int(l)}N" for l in test_loads]
    num_points = 300
    
    # --- Case 1: Alpha Sweep ---
    print("Running Alpha Sweep...")
    alpha_sweep = torch.linspace(-0.3, 0.3, num_points, device=device) # rad
    kappa_zero = torch.zeros_like(alpha_sweep)
    omega_const = torch.full_like(alpha_sweep, 20.0) # ~10 m/s
    
    res_list = []
    for load in test_loads:
        fz_const = torch.full_like(alpha_sweep, load)
        res = tire.forward(kappa_zero, alpha_sweep, fz_const, omega_const)
        res_list.append(preprocess_results(res))
    
    plot_multi_load_results(res_list, fz_labels, np.rad2deg(alpha_sweep.cpu().numpy()), 
                           "Slip Angle (deg)", "Pac02 Pure Lateral (Alpha Sweep)", 
                           "Pac02_Alpha_Sweep.png", save_dir)
    print(f"-> Saved: Pac02_Alpha_Sweep.png")
                           
    # --- Case 2: Kappa Sweep (at Alpha=4deg) ---
    print("Running Kappa Sweep...")
    kappa_sweep = torch.linspace(-1.0, 1.0, num_points, device=device)
    alpha_fixed = torch.full_like(kappa_sweep, np.deg2rad(4.0))
    
    res_list = []
    for load in test_loads:
        fz_const = torch.full_like(kappa_sweep, load)
        res = tire.forward(kappa_sweep, alpha_fixed, fz_const, omega_const)
        res_list.append(preprocess_results(res))
        
    plot_multi_load_results(res_list, fz_labels, kappa_sweep.cpu().numpy(), 
                           "Longitudinal Slip (Kappa)", "Pac02 Combined Slip (Alpha=4deg)", 
                           "Pac02_Kappa_Sweep.png", save_dir)
    print(f"-> Saved: Pac02_Kappa_Sweep.png")
    
    print(f"\nVerification complete. Results saved to {os.path.abspath(save_dir)}")