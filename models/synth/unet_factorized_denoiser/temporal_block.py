from torch import Tensor
from torch.nn import Module, Conv1d, GroupNorm, Dropout, Linear
from torch.nn.functional import silu
from torch.nn.init import zeros_

from models.common import FiLM


class CondSlotTemporalBlock(Module):
	def __init__(self, channels: int, cond_dim: int, kernel_size: int = 3, dilation: int = 1,
				 dropout: float = 0.0, groups: int = 8):
		super().__init__()
		padding = dilation * (kernel_size - 1) // 2
		g = min(groups, channels)

		self.norm1 = GroupNorm(g, channels)
		self.conv1 = Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)

		self.film = FiLM(cond_dim, channels)

		self.norm2 = GroupNorm(g, channels)
		self.drop = Dropout(dropout)
		self.conv2 = Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)

		self.gate_proj = Linear(cond_dim, channels)

		# important: start near identity
		zeros_(self.conv2.weight)
		zeros_(self.conv2.bias)

	def forward(self, x: Tensor, cond: Tensor):
		# x: [B6,C,T], cond: [B6,E]
		h = self.conv1(silu(self.norm1(x)))

		scale, shift = self.film(cond)
		h = h * (1.0 + scale[:, :, None]) + shift[:, :, None]

		h = self.conv2(self.drop(silu(self.norm2(h))))
		gate = self.gate_proj(cond).sigmoid()[:, :, None]

		return x + gate * h