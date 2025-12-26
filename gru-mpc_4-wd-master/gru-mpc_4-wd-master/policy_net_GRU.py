import torch
import torch.nn as nn
import torch.nn.init as init
class PolicyNet(nn.Module):
    def __init__(self, input_state=7, gru_hidden_size=10, gru_num_layers=2, mlp_hidden_dims=[32,64,32], action_dim=1, dropout=0.1):
        super(PolicyNet, self).__init__()   
        self.gru_hidden_size = gru_hidden_size    
        # 多层GRU
        self.gru = nn.GRU(input_size=input_state + 1,
                        hidden_size=gru_hidden_size,
                        num_layers=gru_num_layers,
                        batch_first=True,
                        dropout=dropout if gru_num_layers > 1 else 0)        
        # 构建多层MLP
        mlp_layers = []
        input_dim = gru_hidden_size        
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim        
        mlp_layers.append(nn.Linear(input_dim, action_dim))
        self.mlp = nn.Sequential(*mlp_layers)   
        self.initialize_parameters()    
    def forward(self, state_sequence, valid_lengths):
        batch_size, seq_len, input_state = state_sequence.shape        
        # 创建掩码并拼接，   # 将有效长度扩展为比较的维度
        range_tensor = torch.arange(seq_len).expand(batch_size, seq_len).to(state_sequence.device)
        # 生成掩码: 如果时间步索引 < 有效长度，则为1.0(有效)，否则为0.0(填充)
        mask = (range_tensor < valid_lengths).unsqueeze(-1).float()

        # 2. 将掩码作为额外通道与状态序列拼接 # 此时，GRU就能明确区分每个时间步是真实状态还是填充零了
        augmented_input = torch.cat([state_sequence, mask], dim=-1)        
        # GRU前向传播
        gru_out, _ = self.gru(augmented_input)  
        # 取最后一个有效时间步
        last_valid_output = gru_out[torch.arange(batch_size), valid_lengths - 1]        
        # MLP处理
        return self.mlp(last_valid_output)
    def initialize_parameters(self):
        with torch.no_grad():
            # 1. 初始化GRU的权重
            for name, param in self.gru.named_parameters():
                if 'weight_ih' in name:
                    # 输入到隐藏层的权重：使用正交初始化，适合RNN的激活函数（如tanh、sigmoid）
                    init.orthogonal_(param)
                elif 'weight_hh' in name:
                    # 隐藏层到隐藏层的权重：使用正交初始化，能有效缓解梯度消失/爆炸问题[1](@ref)
                    init.orthogonal_(param)
                elif 'bias' in name:
                    # 偏置项统一初始化为零[1,4](@ref)
                    init.constant_(param, 0)

            # 2. 初始化MLP的权重
            for module in self.mlp:
                if isinstance(module, nn.Linear):
                    # 线性层权重：使用Kaiming初始化，最适合ReLU激活函数[1,8](@ref)
                    init.kaiming_normal_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        init.constant_(module.bias, 0)
if __name__ == '__main__':
    input_state = 5
    gru_hidden_size = 10
    gru_num_layers = 2
    mlp_hidden_dims = [20, 10]
    action_dim = 3
    batch_size = 4
    seq_len = 8
    device='cuda' if torch.cuda.is_available() else 'cpu'
    state_sequence = torch.randn(batch_size, seq_len, input_state).to(device)
    valid_lengths = 0
    policy_net = PolicyNet(input_state, gru_hidden_size, gru_num_layers, mlp_hidden_dims, action_dim).\
        to(device)
    ctr = policy_net(state_sequence, valid_lengths)
