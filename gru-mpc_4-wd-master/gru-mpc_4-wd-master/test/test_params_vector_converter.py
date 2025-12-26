import torch
from torch import nn
from torch.utils._pytree import tree_flatten
from typing import Dict, List
from numpy import prod
from evox.utils import ParamsAndVector

class SimpleNNWithBN(nn.Module):
    def __init__(self, input_size=2, hidden_size=2, output_size=1):
        super(SimpleNNWithBN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),  # 添加BatchNorm1d层
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),  # 添加第二个BatchNorm1d层
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.net(x)

# 创建包含归一化层的MLP模型实例
mlp_model_with_bn = SimpleNNWithBN()

# 检查模型中的参数和缓冲区
print("模型参数结构:")
for name, param in mlp_model_with_bn.named_parameters():
    print(f"{name}: {param.shape}")

print("\n模型缓冲区结构 (归一化层的running_mean和running_var):")
for name, buf in mlp_model_with_bn.named_buffers():
    print(f"{name}: {buf.shape}")

# 初始化ParamsAndVector转换器
params_vector_converter = ParamsAndVector(mlp_model_with_bn)

# 1. 获取模型的当前参数字典
original_params = dict(mlp_model_with_bn.named_parameters())
print("\n原始参数结构:")
for name, param in original_params.items():
    print(f"{name}: {param.shape}")

# 将参数字典转换为单个向量
param_vector = params_vector_converter.to_vector(original_params)
print(f"\n转换后的向量形状: {param_vector.shape}")
print(f"向量总参数量: {param_vector.numel()}")

# 将向量转换回参数字典
reconstructed_params = params_vector_converter.to_params(param_vector)

# 验证转换的正确性
print("\n参数转换验证:")
all_match = True
for name in original_params:
    original = original_params[name]
    reconstructed = reconstructed_params[name]
    match = torch.allclose(original, reconstructed, atol=1e-6)
    print(f"{name}: 匹配结果 - {match}")
    all_match = all_match and match

print(f"\n所有参数完全匹配: {all_match}")

# 关键部分：处理归一化层的缓冲区
print("\n=== 处理归一化层缓冲区的关键步骤 ===")

# 方法1：使用strict=False（推荐）
print("方法1: 使用strict=False加载参数")
try:
    mlp_model_with_bn.load_state_dict(reconstructed_params, strict=False)
    print("✓ 成功加载参数 (strict=False)")
except Exception as e:
    print(f"✗ 加载失败: {e}")

# 方法2：保存并恢复缓冲区
print("\n方法2: 显式保存和恢复缓冲区")
# 保存当前的缓冲区
original_buffers = dict(mlp_model_with_bn.named_buffers())
print("原始缓冲区:")
for name, buf in original_buffers.items():
    print(f"{name}: {buf.shape}")

# 加载参数（这会覆盖参数但保留缓冲区）
mlp_model_with_bn.load_state_dict(reconstructed_params, strict=False)

# 验证缓冲区是否保持不变
current_buffers = dict(mlp_model_with_bn.named_buffers())
buffer_match = True
for name in original_buffers:
    if name in current_buffers:
        match = torch.allclose(original_buffers[name], current_buffers[name], atol=1e-6)
        print(f"缓冲区 {name}: 保持原样 - {match}")
        buffer_match = buffer_match and match

print(f"所有缓冲区保持不变: {buffer_match}")

# 方法3：创建包含缓冲区的完整状态字典
print("\n方法3: 创建完整状态字典（参数+缓冲区）")
full_state_dict = mlp_model_with_bn.state_dict()
print("完整状态字典的键:")
for key in full_state_dict.keys():
    print(f"- {key}")

# 用重建的参数更新完整状态字典
full_state_dict.update(reconstructed_params)
print("\n更新后的状态字典包含:")
print(f"- 参数: {len(reconstructed_params)} 个")
print(f"- 缓冲区: {len([k for k in full_state_dict.keys() if k not in reconstructed_params])} 个")

# 使用完整状态字典加载
mlp_model_with_bn.load_state_dict(full_state_dict)
print("✓ 使用完整状态字典成功加载")

# 最终验证
print("\n=== 最终验证 ===")
# 测试前向传播（使用训练模式）
mlp_model_with_bn.train()
test_input = torch.randn(5, 2)  # batch_size=5, input_size=2
with torch.no_grad():
    output = mlp_model_with_bn(test_input)
print(f"测试输入形状: {test_input.shape}")
print(f"模型输出形状: {output.shape}")
print("✓ 前向传播测试通过")

# 切换到评估模式测试
mlp_model_with_bn.eval()
with torch.no_grad():
    eval_output = mlp_model_with_bn(test_input)
print(f"评估模式输出形状: {eval_output.shape}")
print("✓ 评估模式测试通过")