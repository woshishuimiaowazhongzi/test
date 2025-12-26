import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载数据集
def load_and_prepare_data(device):
    """加载并预处理数据集"""
    # 加载数据
    osqp_dataset = torch.load('./osqp_dataset.pt', map_location=device)
    
    print(f"数据集形状: {osqp_dataset.shape}")
    print(f"数据集类型: {type(osqp_dataset)}")
    
    # 假设数据集形状为 (batch_size, 7, 20)
    # 第二个维度是输入 x0，第三个维度是输出 u
    batch_size, input_dim, output_dim = osqp_dataset.shape
    
    # 分离输入和输出
    # 输入 x0: 取第三维的第 0 个切片（存放了 [kappa, vx, x0(5)] 共 7 个特征）
    x0 = osqp_dataset[:, :, 0]  # 形状: (batch_size, 7)

    # 输出 u: 直接从保存的 osqp_solution.npy 读取，形状 (batch_size, N*Nu)
    u_np = np.load('./osqp_solution.npy')
    if u_np.shape[0] != batch_size:
        raise ValueError(f"osqp_solution.npy 行数 {u_np.shape[0]} 与数据集 {batch_size} 不一致")
    u = torch.from_numpy(u_np).to(device=device, dtype=osqp_dataset.dtype)
    
    print(f"输入 x0 形状: {x0.shape}")
    print(f"输出 u 形状: {u.shape}")
    
    return x0, u, input_dim, output_dim

# 2. 定义神经网络模型
class SimpleNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 32]):
        super(SimpleNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 添加隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# 3. 自定义损失函数
def custom_loss(u_net, u_ref):
    """自定义损失函数: (u_ref - u_net)^2"""
    return torch.mean((u_ref - u_net) ** 2)

# 4. 训练函数
def train_model(x0, u, input_dim, output_dim, device, num_epochs=1000, batch_size=32, learning_rate=0.001):
    """训练模型"""
    
    # 划分训练集和测试集
    dataset_size = x0.shape[0]
    train_size = int(0.8 * dataset_size)
    
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    x0_train, u_train = x0[train_indices], u[train_indices]
    x0_test, u_test = x0[test_indices], u[test_indices]
    
    # 创建数据加载器
    train_dataset = TensorDataset(x0_train, u_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = SimpleNet(input_dim, output_dim).to(device)
    criterion = custom_loss  # 使用自定义损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 存储训练历史
    train_losses = []
    test_losses = []
    
    print("开始训练...")
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        epoch_train_loss = 0.0
        print(batch_u.shape)

        for batch_x0, batch_u in train_loader:
            # 确保 batch 张量在与模型相同的设备上
            batch_x0 = batch_x0.to(device)
            batch_u = batch_u.to(device)
            # 前向传播
            u_net = model(batch_x0)
            loss = criterion(u_net, batch_u)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # 评估模式
        model.eval()
        with torch.no_grad():
            u_net_test = model(x0_test.to(device))
            test_loss = criterion(u_net_test, u_test.to(device))
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        test_losses.append(test_loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Test Loss: {test_loss.item():.6f}')
    
    return model, train_losses, test_losses

# 5. 可视化结果
def plot_results(train_losses, test_losses, model, x0_test, u_test):
    """绘制训练结果"""
    
    # 绘制损失曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.yscale('log')  # 使用对数坐标更好地观察损失下降
    
    # 绘制预测vs真实值
    plt.subplot(1, 3, 2)
    model.eval()
    with torch.no_grad():
        u_pred = model(x0_test)
    
    # 随机选择几个样本进行可视化
    sample_idx = torch.randint(0, u_test.shape[0], (1,)).item()
    
    plt.plot(u_test[sample_idx].cpu().numpy(), 'b-', label='True u', linewidth=2)
    plt.plot(u_pred[sample_idx].cpu().numpy(), 'r--', label='Predicted u', linewidth=2)
    plt.xlabel('Output Dimension')
    plt.ylabel('Value')
    plt.title('Prediction vs True (Sample)')
    plt.legend()
    
    # 绘制损失分布
    plt.subplot(1, 3, 3)
    prediction_errors = (u_pred - u_test).abs().mean(dim=1).cpu().numpy()
    plt.hist(prediction_errors, bins=50, alpha=0.7)
    plt.xlabel('Mean Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Prediction Error Distribution')
    
    plt.tight_layout()
    plt.show()

# 6. 主函数
def main():
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    x0, u, input_dim, output_dim = load_and_prepare_data(device)
    
    # x0/u 已在加载时移动到设备
    
    print(f"输入维度: {input_dim}, 输出维度: {output_dim}")
    print(f"数据集大小: {x0.shape[0]}")
    
    # 训练模型
    model, train_losses, test_losses = train_model(
        x0, u, input_dim, output_dim, device,
        num_epochs=1000,
        batch_size=32,
        learning_rate=0.001
    )
    
    # 绘制结果
    plot_results(train_losses, test_losses, model, x0, u)
    
    # 保存模型
    torch.save(model.state_dict(), 'osqp_model.pth')
    print("模型已保存为 'osqp_model.pth'")

if __name__ == "__main__":
    main()