import torch
import numpy as np
from typing import Dict, Tuple, Optional
import json
import os
import matplotlib.pyplot as plt

from FialaTire_v1 import FialaTire
# ==========================================
# test function
# ==========================================
def preprocess_results(results):
    """
    修复：确保只处理张量，过滤非张量值；增加类型校验
    """
    plot_data = {}
    for key, val in results.items():
        # 关键修复：仅处理PyTorch张量，避免字典/标量导致的cpu()报错
        if isinstance(val, torch.Tensor):
            # 先移到CPU，再转numpy，最后压缩维度
            plot_data[key] = val.cpu().numpy().squeeze()
        else:
            # 非张量值（如标量）直接转换为numpy
            plot_data[key] = np.array(val).squeeze()
    return plot_data

def plot_multi_load_results(results_list, fz_labels, x_data, x_label, title, save_name, save_dir):
    """
    通用绘图：绘制4个子图（fx/fy/my/mz）
    """
    # 改为2行2列的4个子图布局，调整画布尺寸
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    colors = ['r', 'g', 'b', 'c', 'm']  # 颜色循环复用

    # 固定指定4个指标的key和标签（按需求修改）
    key1, label1 = 'fx', 'Longitudinal Force (Fx)'
    key2, label2 = 'fy', 'Lateral Force (Fy)'
    key3, label3 = 'my', 'Rolling Moment (My)'  # 补充My的物理意义标签
    key4, label4 = 'mz', 'Aligning Moment (Mz)'  # 补充Mz的物理意义标签

    # ========== 子图1：Fx ==========
    for idx, (res, label) in enumerate(zip(results_list, fz_labels)):
        if key1 in res:
            ax1.plot(x_data, res[key1], color=colors[idx % len(colors)], linewidth=2, 
                     label=f"{key1.upper()} @ {label}")
    ax1.set_ylabel(f'{label1} [N]', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_title(f'{label1}', fontsize=12)

    # ========== 子图2：Fy ==========
    for idx, (res, label) in enumerate(zip(results_list, fz_labels)):
        if key2 in res:
            ax2.plot(x_data, res[key2], color=colors[idx % len(colors)], linewidth=2, 
                     label=f"{key2.upper()} @ {label}")
    ax2.set_ylabel(f'{label2} [N]', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    ax2.set_title(f'{label2}', fontsize=12)

    # ========== 子图3：My ==========
    for idx, (res, label) in enumerate(zip(results_list, fz_labels)):
        if key3 in res:
            ax3.plot(x_data, res[key3], color=colors[idx % len(colors)], linestyle='--', linewidth=2, 
                     label=f"{key3.upper()} @ {label}")
    ax3.set_xlabel(x_label, fontsize=12)
    ax3.set_ylabel(f'{label3} [Nm]', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')
    ax3.set_title(f'{label3}', fontsize=12)

    # ========== 子图4：Mz ==========
    for idx, (res, label) in enumerate(zip(results_list, fz_labels)):
        if key4 in res:
            ax4.plot(x_data, res[key4], color=colors[idx % len(colors)], linestyle='--', linewidth=2, 
                     label=f"{key4.upper()} @ {label}")
    ax4.set_xlabel(x_label, fontsize=12)
    ax4.set_ylabel(f'{label4} [Nm]', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best')
    ax4.set_title(f'{label4}', fontsize=12)

    # 调整子图间距，避免标签重叠
    plt.tight_layout()
    # 保存图片（确保保存目录存在）
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, save_name), dpi=150, bbox_inches='tight')
    plt.close()

# ==========================================
# 3. 主程序执行逻辑
# ==========================================
if __name__ == "__main__":
    # 配置
    config_path = "./HMMWV_FialaTire.json"
    # 1. 校验配置文件
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在！请确认路径：{config_path}")

    save_dir = "./FialaTire_test_results"
    os.makedirs(save_dir, exist_ok=True)

    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用计算设备: {device}")

    # 2. 初始化轮胎对象
    # tire = FialaTire(config_path)
    tire = FialaTire(config_file=config_path, device=device)
    # 统一变量
    test_loads = [2000.0, 4000.0, 6000.0]
    fz_labels = [f"Fz={int(fz)}N" for fz in test_loads]
    num_points = 300

    # ====================== 工况1：纯侧滑（alpha变化） ======================
    batch_size_1 = 30
    alpha_1 = 0.01 * torch.arange(0, batch_size_1, 1, dtype=torch.float32, device=device)
    alpha_deg_1 = torch.rad2deg(alpha_1).cpu().numpy()
    wheel_omega_1 = torch.full((batch_size_1,), 60.0, dtype=torch.float32, device=device)
    fz_1 = torch.full((batch_size_1,), 2000, dtype=torch.float32, device=device)
    kappa_1 = 0 * torch.ones_like(fz_1, device=device)

    results_1 = tire.forward(kappa_1, alpha_1, fz_1, wheel_omega_1)



   # =================================================================
    # 工况 1：纯侧滑 (Alpha Sweep) -1.0 rad ~ 1.0 rad
    # =================================================================
    print("\n====== Case 1: Pure Side Slip (Alpha Sweep) ======")

    alpha_sweep = torch.linspace(-1.0, 1.0, num_points, device=device)
    alpha_deg = torch.rad2deg(alpha_sweep).cpu().numpy()

    # 纯侧滑时，纵向滑移率 kappa = 0
    kappa_zero = torch.zeros_like(alpha_sweep)
    omega_const = torch.full_like(alpha_sweep, 60.0)

    results_list_alpha = []

    for load in test_loads:
        fz_const = torch.full_like(alpha_sweep, load)
        res = tire.forward(kappa_zero, alpha_sweep, fz_const, omega_const)
        results_list_alpha.append(preprocess_results(res))

    plot_multi_load_results(
        results_list=results_list_alpha,
        fz_labels=fz_labels,
        x_data=alpha_deg,
        x_label='Slip Angle (deg)',
        title='Fiala_Pure Side Slip (Alpha -1.0~1.0 rad)',
        save_name='Fiala_Case1_Alpha_Sweep.png',
        save_dir=save_dir
    )
    print(f"-> Saved: Fiala_Case1_Alpha_Sweep.png")

    print("=== 纯侧滑工况 Tire Results (res) ===")
    for key, val in res.items():
        val_np = val.cpu().numpy().squeeze()
        # 仅打印前3个值示例
        print(f"{key.upper()}: {val_np[:3]} ...")
   # =================================================================
    # 工况 2：纵滑 (Kappa Sweep) -1.0 ~ 1.0
    # 混合不同的固定侧偏角 Alpha = [0, 5, 10, 15, 25] deg
    # =================================================================
    print("\n====== Case 2: Longitudinal Slip (Kappa Sweep) with Mixed Alpha ======")

    test_alphas_deg = [0, 5, 10, 15, 25]
    kappa_sweep = torch.linspace(-1.0, 1.0, num_points, device=device)
    kappa_numpy = kappa_sweep.cpu().numpy()

    for alpha_val in test_alphas_deg:
        print(f"Processing Alpha = {alpha_val}° ...")

        # 固定侧偏角
        alpha_rad = np.deg2rad(alpha_val)
        alpha_const = torch.full_like(kappa_sweep, alpha_rad)

        results_list_kappa = []

        for load in test_loads:
            fz_const = torch.full_like(kappa_sweep, load)
            res = tire.forward(kappa_sweep, alpha_const, fz_const, omega_const)
            results_list_kappa.append(preprocess_results(res))

        # 绘图：主要看 Fx，次要看 Fy（Fy会因为纵滑增加而减小）
        plot_multi_load_results(
            results_list=results_list_kappa,
            fz_labels=fz_labels,
            x_data=kappa_numpy,
            x_label='Longitudinal Slip Ratio (kappa)',
            title=f'Fiala_Kappa Sweep at Fixed Alpha={alpha_val}°',
            save_name=f'Fiala_Case2_Kappa_Sweep_Alpha_{alpha_val}deg.png',
            save_dir=save_dir
        )
        print(f"  -> Saved: Fiala_Case2_Kappa_Sweep_Alpha_{alpha_val}deg.png")

    print(f"\nAll results saved in: {os.path.abspath(save_dir)}")

    for key, val in res.items():
        val_np = val.cpu().numpy().squeeze()
        print(f"{key.upper()}: {val_np[:3]} ...")

  
  