import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# 导入模型文件
from TMSimpleTire_Model import TMsimpleTire

# --------------------------------------------------------
# 配置区
# --------------------------------------------------------
SAVE_DIR = "./tmsimple_test_results"
JSON_PATH = "HMMWV_TMsimpleTire.json"

# 设置测试设备
TEST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_dummy_json(path):
    """ 创建测试用的轮胎参数文件 (如果不存在) """
    if not os.path.exists(path):
        print(f"Creating dummy JSON at {path}")
        data = {
            "Name": "HMMWV TMsimple Default",
            "Design": {
                "Mass [kg]": 37.6,
                "Unloaded Radius [m]": 0.4699,
                "Width [m]": 0.3175,
                "Rim Radius [m]": 0.2095
            },
            "Coefficient of Friction": 0.8,
            "Rolling Resistance Coefficient": 0.015,
            "Load Index": 108,
            "Vehicle Type": "Truck"
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

def preprocess_results(results):
    return {k: v.cpu().numpy().squeeze() for k, v in results.items()}

# ==============================================================================
# 测试 1: Alpha Sweep (纯侧偏工况)
# ==============================================================================
def run_alpha_sweep(tire, loads):
    print("\n[Test 1] Running Alpha Sweep...")
    num_points = 300
    alpha_deg = torch.linspace(-20, 20, num_points, device=TEST_DEVICE)
    alpha_rad = torch.deg2rad(alpha_deg)
    
    kappa_zero = torch.zeros_like(alpha_rad)
    omega_const = torch.full_like(alpha_rad, 20.0) # 正常行驶
    vx_const = torch.full_like(alpha_rad, 10.0)    # 10m/s
    
    plt.figure(figsize=(10, 6))
    
    for fz_val in loads:
        fz = torch.full_like(alpha_rad, fz_val)
        res = tire.forward(kappa_zero, alpha_rad, fz, omega_const, vx=vx_const)
        res_np = preprocess_results(res)
        
        plt.plot(alpha_deg.cpu().numpy(), res_np['fy'], linewidth=2, label=f'Fz={int(fz_val)}N')
        
    plt.title('Lateral Force vs Slip Angle (Alpha Sweep)')
    plt.xlabel('Slip Angle [deg]')
    plt.ylabel('Lateral Force Fy [N]')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'Test1_Alpha_Sweep.png'))
    plt.close()
    print("-> Saved Test1_Alpha_Sweep.png")

# ==============================================================================
# 测试 2: Kappa Sweep (纯纵向滑移工况)
# ==============================================================================
def run_kappa_sweep(tire, loads):
    print("\n[Test 2] Running Kappa Sweep...")
    num_points = 300
    kappa = torch.linspace(-1.0, 1.0, num_points, device=TEST_DEVICE)
    
    alpha_zero = torch.zeros_like(kappa)
    omega_const = torch.full_like(kappa, 20.0)
    vx_const = torch.full_like(kappa, 10.0)
    
    plt.figure(figsize=(10, 6))
    
    for fz_val in loads:
        fz = torch.full_like(kappa, fz_val)
        res = tire.forward(kappa, alpha_zero, fz, omega_const, vx=vx_const)
        res_np = preprocess_results(res)
        
        plt.plot(kappa.cpu().numpy(), res_np['fx'], linewidth=2, label=f'Fz={int(fz_val)}N')
        
    plt.title('Longitudinal Force vs Slip Ratio (Kappa Sweep)')
    plt.xlabel('Slip Ratio Kappa')
    plt.ylabel('Longitudinal Force Fx [N]')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'Test2_Kappa_Sweep.png'))
    plt.close()
    print("-> Saved Test2_Kappa_Sweep.png")

# ==============================================================================
# 测试 3: Standstill / Low Speed (vx=0, 零速测试)
# ==============================================================================
def run_standstill_test(tire, load_val):
    print("\n[Test 3] Running Standstill Test (vx=0, Spin-up)...")
    
    # 模拟设置
    dt = 0.001       # 1ms
    duration = 2.0   # 2秒
    steps = int(duration / dt)
    
    # 输入信号构造
    fz = torch.full((steps,), load_val, device=TEST_DEVICE)
    # 关键：车速严格为 0
    vx = torch.zeros(steps, device=TEST_DEVICE)
    vy = torch.zeros(steps, device=TEST_DEVICE)
    alpha = torch.zeros(steps, device=TEST_DEVICE)
    kappa = torch.zeros(steps, device=TEST_DEVICE) # 低速时 kappa 模型不使用
    
    # 动作：车轮转速从 0 线性增加到 5 rad/s (模拟起步)
    omega = torch.linspace(0, 5.0, steps, device=TEST_DEVICE)
    
    # 记录结果
    fx_history = []
    
    # 重置模型内部积分状态
    tire.brx = None 
    tire.bry = None
    
    # 时间步循环
    for i in range(steps):
        # 传入单步数据
        res = tire.forward(
            kappa=kappa[i:i+1],
            alpha=alpha[i:i+1],
            fz=fz[i:i+1],
            omega=omega[i:i+1],
            vx=vx[i:i+1],
            vy=vy[i:i+1],
            step=dt
        )
        fx_history.append(res['fx'].item())
        
    # 绘图
    time_axis = np.linspace(0, duration, steps)
    omega_np = omega.cpu().numpy()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Wheel Speed Omega [rad/s]', color='blue')
    ax1.plot(time_axis, omega_np, color='blue', linestyle='--', label='Omega')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Longitudinal Force Fx [N]', color='red')
    ax2.plot(time_axis, fx_history, color='red', linewidth=2, label='Fx (Static/Dahl)')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title(f'Standstill Force Generation (vx=0, Fz={int(load_val)}N)')
    plt.grid(True, alpha=0.3)
    
    # 结果说明
    plt.figtext(0.15, 0.8, "Force builds up as internal bristles deflect (Dahl Model)", 
                fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.savefig(os.path.join(SAVE_DIR, 'Test3_Standstill_vx0.png'))
    plt.close()
    print("-> Saved Test3_Standstill_vx0.png")

# ==============================================================================
# 主执行入口
# ==============================================================================
if __name__ == "__main__":
    # 1. 准备环境
    os.makedirs(SAVE_DIR, exist_ok=True)
    create_dummy_json(JSON_PATH)
    
    # 2. 实例化模型
    tire = TMsimpleTire(JSON_PATH, device=TEST_DEVICE)
    print(f"Model Loaded: {tire.name} on {TEST_DEVICE}")
    
    # 3. 设置测试载荷
    test_loads = [4000.0, 8000.0]
    
    # 4. 运行所有测试
    run_alpha_sweep(tire, test_loads)
    run_kappa_sweep(tire, test_loads)
    run_standstill_test(tire, 8000.0)
    
    print(f"\nAll tests completed. Results saved in: {os.path.abspath(SAVE_DIR)}")