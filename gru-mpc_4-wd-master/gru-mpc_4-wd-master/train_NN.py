import torch
from DynCloseloopWithNet import VehicleDynCloseloop
from init_mpc import init_MPC
from init_vehicle_sim import P_Vehicle
from init_steer_actuator_sim import init_steer_actuator_sim
from datetime import datetime
from gen_vehicle_dataset import create_dataloader,gen_vehicle_dataset,sampling_space_change,sampling_noise_space_change
from gen_dataset_utility import calc_interval,batch_add_noise
from cal_loss import loss_function
from cal_KxKu_batch import batch_cal_KxKu
import numpy as np
import pandas as pd
p_vehicle = P_Vehicle(device='cuda' if torch.cuda.is_available() else 'cpu')
init_steer_actuator_sim(p_vehicle)       
mpc_config = init_MPC(p_vehicle)
dyn_cl = VehicleDynCloseloop(mpc_config.Ts,mpc_config.N)

optimizer = torch.optim.AdamW(dyn_cl.policy_net.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-8
    )
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#    optimizer, T_max=10, eta_min=1e-6)          
num_epochs = 1000  # 增加训练最大轮数
patience = 50     # 减少早停耐心值:
min_delta = 1e-4  # 最小改善阈值
best_loss = float('inf')
counter = 0


num_sample_per_dim = [8,8,8,8,8,8,8] # 8**7 = 2097152
total_sample_num = np.prod(num_sample_per_dim)

batch_size = int(32768)

batch_num = int(total_sample_num/batch_size)
print(f'batch_num: {batch_num}')

train_data,bounds = gen_vehicle_dataset(p_vehicle,num_sample_per_dim,with_noise=False,reserve_boundary=True)
#dara_org加入噪声
interval_list = calc_interval(num_sample_per_dim,bounds,reserve_boundary=True)
#train_data = sampling_noise_space_change(train_data,p_vehicle,interval_list)
train_loader = create_dataloader(train_data,batch_size,True)

dev_data,bounds_dev = gen_vehicle_dataset(p_vehicle,num_sample_per_dim,with_noise=False,reserve_boundary=True)
dev_change = sampling_noise_space_change(dev_data,p_vehicle,interval_list)
# dev_data
np.save('dev_data.npy', dev_change.cpu().numpy())
dev_loader = create_dataloader(dev_change,batch_size,shuffle=False)
# Training loop
train_losses = []
dev_losses = []
#dyn_cl.policy_net.load_policy_net('best_model_checkpoint.pth')
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始训练...")
#dyn_cl.load_policy_net('best_model_checkpoint.pth')
# 训练循环
for epoch in range(num_epochs):
    dyn_cl.policy_net.train()
    running_loss = 0.0    
    for batch_idx, batch in enumerate(train_loader):
        #print('batch_idx:',batch_idx)
        optimizer.zero_grad()
        batch_org = batch[0]
        batch_size = batch_org.size(0)  # 获取当前批次样本数
        #batch_features = batch_org#(batch_org,p_vehicle,interval_list)
        batch_features = sampling_noise_space_change(batch_org,p_vehicle,interval_list)
        kappa, vx_bar = batch_features[:,0:1],batch_features[:,1:2]
        x0 = batch_features[:,2:7]
        dyn_cl.set_system_params(kappa, vx_bar)
        x_batch, u_batch,t_batch = dyn_cl.solve(x0)
        Kx, Ku = batch_cal_KxKu(vx_bar, p_vehicle)
        x_ref = Kx * kappa#torch.zeros(batch_size,5,device=vx_bar.device)#
        u_ref =  Ku * kappa#torch.zeros(batch_size,1,device=vx_bar.device)#
        #loss_function(mpc_config, x_ref,u_ref,x_batch, u_batch)      
        loss = loss_function(mpc_config, x_ref,u_ref,x_batch, u_batch)        
        loss.backward()
        totall_norm = torch.nn.utils.clip_grad_norm_(dyn_cl.policy_net.parameters(), 300)  # 添加梯度裁剪
        #print(f"Batch {batch_idx}, Total Norm: {totall_norm}")
        optimizer.step()
        running_loss += loss.item()   
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f},total_norm: {totall_norm:.4f}")     
    avg_train_loss = running_loss / len(train_loader)  # 基于样本数计算平均损失 
 # Validation
    # dyn_cl.policy_net.eval()  # 设置模型为评估模式
    # running_loss = 0.0  # 初始化损失累加器    
    # with torch.no_grad():  # 禁用梯度计算
    #     for batch in dev_loader:  # 遍历验证集的所有批次
    #         batch_features = batch[0]
    #         batch_size_current = batch_features.size(0)  # 获取当前批次样本数
    #         kappa, vx_bar = batch_features[:,0:1],batch_features[:,1:2]
    #         x0 = batch_features[:,2:7]
    #         dyn_cl.set_system_params(kappa, vx_bar)
    #         x_batch, u_batch,t_batch = dyn_cl.solve(x0)
    #         Kx, Ku = batch_cal_KxKu(vx_bar, p_vehicle)
    #         x_ref = Kx * kappa#torch.zeros(batch_size,5,device=vx_bar.device)#
    #         u_ref =  Ku * kappa#torch.zeros(batch_size,1,device=vx_bar.device)#
    #         loss = loss_function(mpc_config, x_ref,u_ref,x_batch, u_batch)                          
    #         running_loss += loss.item()   
    #     avg_dev_loss = running_loss / len(dev_loader)  # 基于样本数计算平均损失
    current_lr = optimizer.param_groups[0]['lr']
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch}, Train Loss: {avg_train_loss:.4f},LR: {current_lr:.2e}")#, Dev Loss: {avg_dev_loss:.4f}

    # 学习率调度
    scheduler.step(avg_train_loss)
    #scheduler.step()
    best_loss = 10**9
    if avg_train_loss < best_loss - min_delta:  # 当前损失优于最佳损失，保存当前损失为best_model_state
        best_loss = avg_train_loss
        counter = 0
        best_model_state = dyn_cl.policy_net.state_dict()
        # 保存最佳模型检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': dyn_cl.policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'train_loss': avg_train_loss,
            'dev_loss': avg_train_loss
        }, 'best_model_checkpoint.pth')

    





