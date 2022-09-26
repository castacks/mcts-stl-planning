import torch
from torch import nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
	def __init__(self, chomp_size):
		super(Chomp1d, self).__init__()
		self.chomp_size = chomp_size

	def forward(self, x):
		return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
	def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
		super(TemporalBlock, self).__init__()
		self.conv1 = (nn.Conv1d(n_inputs, n_outputs, kernel_size,
										   stride=stride, padding=padding, dilation=dilation))
		self.chomp1 = Chomp1d(padding)
		self.bn1 = nn.BatchNorm1d(num_features=n_outputs)
		self.relu1 = nn.PReLU(num_parameters=n_outputs)
		self.dropout1 = nn.Dropout(dropout)

		self.conv2 = (nn.Conv1d(n_outputs, n_outputs, kernel_size,
										   stride=stride, padding=padding, dilation=dilation))
		self.chomp2 = Chomp1d(padding)
		self.bn2 = nn.BatchNorm1d(num_features=n_outputs)
		self.relu2 = nn.PReLU(num_parameters=n_outputs)
		self.layer2= nn.LayerNorm([n_outputs,20])
		self.dropout2 = nn.Dropout(dropout)

		self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1,self.relu1, self.dropout1,
								 self.conv2, self.chomp2, self.bn2,self.relu2, self.dropout2)
		self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
		self.relu = nn.PReLU(num_parameters=n_outputs)
#         self.tanh = nn.Tanh()
		self.init_weights()

	def init_weights(self):
		self.conv1.weight.data.normal_(0, 0.1)
		self.conv2.weight.data.normal_(0, 0.1)
		if self.downsample is not None:
			self.downsample.weight.data.normal_(0, 0.01)

	def forward(self, x):
		out = self.net(x)
		res = x if self.downsample is None else self.downsample(x)
		return self.relu(out + res)


class TemporalConvNet(nn.Module):
	def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
		super(TemporalConvNet, self).__init__()
		layers = []
		num_levels = len(num_channels)
#         print("num_levels",num_levels)
		for i in range(num_levels):
			dilation_size = 2 ** i
			in_channels = num_inputs if i == 0 else num_channels[i-1]
#             print("in_channels",in_channels)
			out_channels = num_channels[i]
			layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
									 padding=(kernel_size-1) * dilation_size, dropout=dropout)]

		self.network = nn.Sequential(*layers)

	def forward(self, x):
		return self.network(x)