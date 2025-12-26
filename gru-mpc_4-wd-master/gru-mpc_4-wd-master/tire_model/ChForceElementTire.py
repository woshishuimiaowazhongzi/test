import torch
import sys
import os
# 获取项目根目录路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)  # 将根目录加入模块搜索路径
from tire_model.ChTire import ChTire
class ChForceElementTire(ChTire):
    def __init__(self, name, device='cpu'):
        super().__init__(name, device)
        self.data_normal_force = None