import torch.nn as nn


class DQN(nn.Module):
	def __init__(self, out_length):
		super(DQN, self).__init__()
		self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
		self.conv2 = nn.Conv2d(16, 16, 3, 1, 1)
		self.linear = nn.Linear(out_length * 16, out_length * 4)
		self.out = nn.Linear(out_length * 4, out_length)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		x = x.view(x.size(0), -1)
		out = self.out(self.relu(self.linear(x)))
		return out
