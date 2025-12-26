import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from supervised_learning_net import MLP
from datetime import datetime


# 固定随机种子提高复现性
# torch.manual_seed(42)
# np.random.seed(42)


# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# osqp数据集
osqp_dataset = torch.load('./osqp_dataset.pt').to(device)


# 现在 osqp_dataset 的最后一维为 N*Nu+1，其中索引0保存原始特征，1: 保存控制序列
batch_size_total, input_dim, last_dim = osqp_dataset.shape
u_dim = 1
print(f'数据集形状: {osqp_dataset.shape} -> (batch_size_total={batch_size_total}, input_dim={input_dim}, last_dim={last_dim}, u_dim={u_dim})')


x = osqp_dataset[:, :, 0].contiguous()              # (B, input_dim)
u= osqp_dataset[:, 0, 1:2].contiguous()             # (B, u_dim)



# DataLoader
batch_size = 2**14
dataset = TensorDataset(x, u)

# 全量作为训练集
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、优化器
model = MLP(input_dim, u_dim).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )

num_epochs = 200
best_train = float('inf')
best_state = None

print(f'开始训练: epochs={num_epochs}, batch_size={batch_size}, device={device}')
for epoch in range(1, num_epochs + 1):
	# 训练
	model.train()
	total_train = 0.0
	n_train = 0
	for x0, u_star in train_loader:
		optimizer.zero_grad()
		u_net = model(x0)
		# 每个样本：先在特征维度求和平方误差，再对 batch 取平均
		# 等价于 (u_net - u_ref).pow(2).sum() / batch_size
		diff = (u_net - u_star)*100
		loss = diff.pow(2).mean()
		if torch.isnan(loss) or torch.isinf(loss):
			continue
		loss.backward()
		nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		optimizer.step()
		total_train += loss.item() * x0.size(0)
		n_train += x0.size(0)
	avg_train = total_train / max(n_train, 1)

	# 学习率调度基于训练损失
	scheduler.step(avg_train)

	# 保存最佳训练损失的模型
	if avg_train < best_train:
		best_train = avg_train
		# 深拷贝一份最佳权重，避免后续训练过程中被覆盖
		best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
		# 训练过程中立刻保存为“当前最佳”权重，便于随时中断和使用
		torch.save(model.state_dict(), './supervised_osqp_mlp.pth')
		print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | New best saved at epoch {epoch}: train {best_train:.6f} -> supervised_osqp_mlp.pth")

	timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	print(f'{timestamp} | Epoch {epoch:03d} | train {avg_train:.6f} | lr {optimizer.param_groups[0]["lr"]:.2e}')

# 保存最优模型
if best_state is not None:
	model.load_state_dict(best_state)
torch.save(model.state_dict(), './supervised_osqp_mlp.pth')
print('训练完成，已保存权重到 supervised_osqp_mlp.pth')

model.eval()
x0_list = [0.01, 10.0, 0.1, 0.1, 0.1, 0.1, 0.1]  # [kappa, vx, x0(5)]
x0 = torch.tensor(x0_list, dtype=torch.float32, device=device).unsqueeze(0)  # [1, 7]
with torch.no_grad():
	u_net = model(x0)
	print('示例预测输出形状:', u_net.shape)
	print(u_net[0])
