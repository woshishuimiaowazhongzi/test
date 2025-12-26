import numpy as np
import matplotlib.pyplot as plt

# 加载数据
x_torchode = np.load('result/x_net.npy')
x_osqp = np.load('result/x_osqp.npy')

# 绘制结果
N = x_torchode.shape[0]  # 时间步数
Nx = x_torchode.shape[1]  # 状态变量数

# 时间步
steps = np.arange(N)

# 创建5个子图（1列5行）
fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

# 绘制每个状态变量的对比曲线
for i in range(5):
    ax = axes[i]
    ax.plot(steps, x_torchode[:, i], label=f'net x[{i}]', color='blue')
    ax.plot(steps, x_osqp[:, i], label=f'osqp x[{i}]', color='red')
    ax.set_ylabel(f'State x[{i}]')
    ax.grid(True)
    ax.legend(loc='best')

# 设置共享的x轴标签
axes[-1].set_xlabel('Time Steps')

# 调整布局
fig.tight_layout(rect=[0, 0.03, 1, 0.97])

# 显示图形
plt.show()
