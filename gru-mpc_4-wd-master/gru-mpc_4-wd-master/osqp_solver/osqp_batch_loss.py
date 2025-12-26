import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init_mpc import init_MPC
from init_mpc import mpc_config_to_numpy
from init_vehicle_sim import P_Vehicle
from init_steer_actuator_sim import init_steer_actuator_sim
from mpc2qp import formulate_qp_problem
from mpc2qp import zoh_discrete
import numpy as np
import torch
import matplotlib.pyplot as plt
from osqp_solver import solve_mpc_osqp
from osqp_solver import cal_osqp_mpc_loss
from time import time  # 导入time函数用于计时


vehicle = P_Vehicle(device='cuda' if torch.cuda.is_available() else 'cpu') 
init_steer_actuator_sim(vehicle) 
p_vehicle = vehicle.to_numpy() # 把车辆参数转化为numpy版本
mpc = init_MPC(p_vehicle)
mpc_config = mpc_config_to_numpy(mpc) # 把mpc初始化参数转化为numpy版本
# 加载上一层目录的npy数据
data = np.load('./dev_data.npy')
print(data.shape)

# 滑动平均数值
avg_mpc_loss = 0.0
# 时间统计变量
total_time = 0.0
start_batch_time = time()  # 记录批次开始时间

# 使用列表暂存每次的 [u_0..u_{N-1}] （仅保存控制量u，不保存状态x）
u_rows = []
# mpc相关参数
N = mpc_config.N
Nx = mpc_config.Nx 
Nu = mpc_config.Nu



for index in range(data.shape[0]):
    kappa = data[index,0]
    vx = data[index,1]    
    x0 = data[index,2:7]
    
    # 构建QP问题
    H, q, Aeq, beq = formulate_qp_problem(mpc_config, p_vehicle, kappa, vx, x0)
    # osqp求解器返回的损失和解
    osqp_loss, solution, status = solve_mpc_osqp(H, q, Aeq, beq)
    # 从 solution 提取控制量 u0到u_N-1
    u_flat = []
    for k in range(N):
        x_idx = k * (Nx + Nu)
        u_idx = x_idx + Nx
        u_k = solution[u_idx:u_idx + Nu]
        # 追加本时刻控制到扁平列表
        u_flat.extend(u_k.tolist())
    row = u_flat
    u_rows.append(row)
    
    if status != 'solved':
        print(index,'solve failed and ','status:', status)
    # 手动计算osqp的损失和mpc问题的损失
    osqp_loss_manual, mpc_loss_manual = cal_osqp_mpc_loss(solution, mpc_config, p_vehicle, kappa, vx)
    avg_mpc_loss = (avg_mpc_loss * index + mpc_loss_manual) / (index + 1)
    
    #每1000次输出一次平均损失和时间消耗
    if (index + 1) % 1000 == 0:
        end_batch_time = time()
        batch_time = end_batch_time - start_batch_time
        avg_time_per_iteration = batch_time / 1000
        
        print('index:', index, 'avg_mpc_loss:', avg_mpc_loss, 
              'avg_time_per_iteration:', avg_time_per_iteration, 'seconds')
        
        # 重置批次计时
        start_batch_time = time()
    elif index == 0:
        # 单独处理第一次迭代
        print('index:', index, 'avg_mpc_loss:', avg_mpc_loss)


data_tensor = torch.from_numpy(data) # 将data转化为torch张量
# 注意：若直接 osqp_dataset[:,0,:] = osqp_solution，会覆盖 osqp_dataset[:,:,0] 的第0列（索引 [*,0,0]）。
# 为避免覆盖，把最后一维扩展为 N*Nu + 1：索引0专门存放原始特征(data)，索引1: 存放控制序列(u)。
osqp_dataset = torch.zeros((data_tensor.shape[0], 7, N*Nu + 1)) # 定义空的数据集
osqp_dataset[:, :, 0] = data_tensor # 第0个“通道”用于保存原始特征 [kappa, vx, x0...]

osqp_solution = np.asarray(u_rows) # 循环结束后一次性构建并保存 osqp_solution
osqp_solution = torch.from_numpy(osqp_solution) # 将osqp_solution转化为torch张量

# 将控制序列放入不与 data 冲突的切片位置 1:
osqp_dataset[:, 0, 1:] = osqp_solution
print('正在保存osqp数据集')
torch.save(osqp_dataset, 'osqp_dataset.pt')
print('osqp_dataset 保存完成')


# # 这里应当与 data_tensor 完全一致：
# print(osqp_dataset[:, :, 0])

# print(data_tensor)

# print(osqp_dataset[:, 0, 1:])
# print(osqp_solution)
# print(osqp_dataset.shape)



