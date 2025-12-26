import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    def __init__(self, input_dim=7, output_dim=1, hidden_dims=[32,64,32], 
                 dropout_rate=0.1, use_residual=True):
        """
        增强型策略网络 - 专为车辆控制优化
        
        参数:
            input_dim: 输入维度 (默认7: [kappa, Vx_bar, 5个状态变量])
            output_dim: 输出维度 (默认1: 控制量)
            hidden_dims: 隐藏层维度列表 [256, 128, 64]
            dropout_rate: Dropout比率，防止过拟合
            use_residual: 是否使用残差连接
        """
        super(PolicyNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.use_residual = use_residual
        
        # 构建主干网络
        layers = []
        prev_dim = input_dim
        #layers.append(nn.LayerNorm(input_dim))#归一化输入特征
        for i, hidden_dim in enumerate(hidden_dims):
            # 线性层
            layers.append(nn.Linear(prev_dim, hidden_dim))            
            # 层归一化（除最后一层外）
            #if i < len(hidden_dims) - 1:
            #   layers.append(nn.LayerNorm(hidden_dim))            
            # Mish激活函数 - 比ReLU更平滑，训练更稳定
            layers.append(nn.Mish(inplace=True))
            #layers.append(nn.ReLU())
            
            # Dropout正则化（除最后一层外）
            if i < len(hidden_dims) - 1 and dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # 输出层 - 使用Tanh将输出限制在合理范围
        self.output_layer = nn.Linear(prev_dim, output_dim)
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Xavier均匀初始化，保持梯度稳定"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    # he初始化
    def _init_he_weights(self, module):
        """He初始化，保持梯度稳定"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        """
        前向传播        
        参数:
            x: 输入张量 [batch_size, input_dim]            
        返回:
            控制动作: [batch_size, output_dim]
        """   
        # 主干网络特征提取
        features = self.backbone(x)        
        # 输出层
        output = self.output_layer(features)
        #x0 = torch.cat([x[:,0:1],x[:,2:]],dim=1)
        #output = torch.sum(output*x0,dim=1,keepdim=True)
        #output = torch.clamp(output, min=-1, max=1)
        return output
    def load_policy_net(self,model_path):
        checkpoint = torch.load(model_path)                
        self.load_state_dict(checkpoint['model_state_dict']) 
        self.eval() 

if __name__ == '__main__':
    policy_net = PolicyNet()
    policy_net.load_policy_net('best_model_checkpoint.pth')
    batch_size = 3
    x = torch.randn(batch_size, 7)
    output = policy_net(x)

    print(output.shape)
