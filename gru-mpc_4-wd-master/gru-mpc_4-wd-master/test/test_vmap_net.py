import torch
import torch.nn as nn
from torch.func import vmap, functional_call, stack_module_state
import copy

# 1. 定义策略网络结构
class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.net(x)

# 2. 创建策略网络种群
pop_size = 10
input_size = 8
hidden_size = 128
output_size = 4
models = [PolicyNet(input_size, hidden_size, output_size) for _ in range(pop_size)]
# 3. 堆叠参数和缓冲区
params, buffers = stack_module_state(models)

# 4. 创建无状态模型函数
base_model = copy.deepcopy(models[0])
base_model = base_model.to('meta')
def stateless_model(params, buffers, x):
    return functional_call(base_model, (params, buffers), (x,))

# 5. 使用vmap进行并行评估
batch_size = 32
eval_data = torch.randn(batch_size, input_size)
batched_predict = vmap(stateless_model, in_dims=(0, 0, None))  # 对params和buffers的第0维进行映射，输入x不映射

with torch.no_grad():
    all_outputs = batched_predict(params, buffers, eval_data)

print(f"并行评估输出形状: {all_outputs.shape}")  # 应为 (pop_size, batch_size, output_size)