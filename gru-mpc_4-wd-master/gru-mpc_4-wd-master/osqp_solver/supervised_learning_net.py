import torch
import torch.nn as nn

#  MLP，输入维度 input_dim，输出维度 output_dim
class MLP(nn.Module):
	def __init__(self, in_dim: int, out_dim: int, hidden: list[int] = [128, 128]):
		super().__init__()
		layers: list[nn.Module] = []
		prev = in_dim
		for h in hidden:
			layers += [nn.Linear(prev, h), nn.ReLU()]
			prev = h
		layers += [nn.Linear(prev, out_dim)]
		self.net = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)