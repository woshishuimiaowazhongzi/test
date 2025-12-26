import torch
from torch import vmap

# 假设这是您原有的支持batch_size的函数
def your_model(x):
    """
    您的原有函数，输入x形状为[batch_size, ...]，输出为[batch_size, ...]
    """
    # 您的模型逻辑
    return x # 简化示例

# 创建vmap包装的函数
vmap_model = vmap(your_model, in_dims=0, out_dims=0)

# 使用示例
batch_size, pop_size = 32, 10
feature_shape = (4,)  # 假设是图像数据

# 原始输入（单一批次）：形状为[batch_size, ...]
single_batch_input = torch.randn(batch_size, *feature_shape)

# 新输入（添加pop_size维度）：形状为[pop_size, batch_size, ...]
batched_input = torch.randn(pop_size, batch_size, *feature_shape)

# 使用vmap处理：自动在pop_size维度上向量化
output = vmap_model(batched_input)
print(f"输入形状: {batched_input.shape}")
print(f"输出形状: {output.shape}")  # 将是[pop_size, batch_size, ...]