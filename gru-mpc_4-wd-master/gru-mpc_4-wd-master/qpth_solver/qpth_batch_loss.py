import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init_mpc import init_MPC
from init_vehicle_sim import P_Vehicle
from init_steer_actuator_sim import init_steer_actuator_sim
from qpth_solver import solve_mpc_qpth
from qpth_solver import cal_qpth_mpc_loss_batch
from mpc2qp_batch import formulate_qp_problem_batch
import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time  # 导入time函数用于计时

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
p_vehicle = P_Vehicle(device=device)
init_steer_actuator_sim(p_vehicle)
mpc_config = init_MPC(p_vehicle)

data = np.load('./dev_data.npy')
data = torch.from_numpy(data).to(device) # numpy格式
# 新建数据集，用于存放不同批次QPth的决策变量\
# 把预测时域内参考值kappa和vx，以及优化变量solution存到qpth_solution中前两个中
kappa_batch = data[:, 0]
vx_batch = data[:, 1]
qpth_solution = torch.zeros(data.shape[0], mpc_config.N * (mpc_config.Nx + mpc_config.Nu) + mpc_config.Nx + 2, device=device) # 每个batch中，kappa和vx的索引为0和1
# 把kappa_batch和vx_batch分别存到qpth_solution中
qpth_solution[:, 0] = kappa_batch
qpth_solution[:, 1] = vx_batch
# print(qpth_solution.shape)
# print(data.shape)
# 滑动平均数值
avg_mpc_loss = 0.0

# 数据集的batch_size为1000000
# 使用qpth每100次批次计算一次MPC (减小批次大小以避免内存溢出)
batch_size = 1000
total_batches = data.shape[0] // batch_size
processed_samples = 0

for batch_index in range(total_batches):
    
    # 开始计时
    end_batch_time = time()
    start_idx = batch_index * batch_size
    end_idx = (batch_index + 1) * batch_size
    data_batch = data[start_idx:end_idx, :]
    # 构建批次QP问题
    H_batch, q_batch, Aeq_batch, beq_batch, G_ieq_batch, h_ieq_batch = formulate_qp_problem_batch(mpc_config, p_vehicle, data_batch)
    # QPTH求解器求解
    solution = solve_mpc_qpth(H_batch, q_batch, Aeq_batch, beq_batch, G_ieq_batch, h_ieq_batch)
    # 储存在qpth_solution中的solution索引从2开始
    qpth_solution[start_idx:end_idx, 2:] = solution
    # 计算损失
    qpth_loss, mpc_loss = cal_qpth_mpc_loss_batch(solution, mpc_config, p_vehicle, data_batch)
    # 计算当前批次的总损失
    batch_total_loss = torch.sum(mpc_loss).item()
    current_batch_size = mpc_loss.shape[0]  # 最后一个批次可能不足batch_size
    
    # 更新全局平均损失
    avg_mpc_loss = (avg_mpc_loss * processed_samples + batch_total_loss) / (processed_samples + current_batch_size)
    processed_samples += current_batch_size
    
    # 计算时间消耗
    start_batch_time = time()
    batch_time = end_batch_time - start_batch_time
    avg_time_per_iteration = batch_time / current_batch_size
    
    print(f'batch_index: {batch_index}, processed_samples: {processed_samples}, '
          f'avg_mpc_loss: {avg_mpc_loss}, '
          f'avg_time_per_iteration: {avg_time_per_iteration} seconds')
    
    # 重置批次计时
    start_batch_time = time()

# 保存qpth_solution
qpth_solution = qpth_solution.cpu().numpy()
np.save('qpth_solution.npy', qpth_solution)
print("qpth_solution保存完成")

    