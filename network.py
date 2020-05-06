import torch
from torch import nn, optim
from torch.nn import functional as F

NETWORK_INPUT_SIZE = 100
NETWORK_OUTPUT_SIZE = 3


class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2)
		self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2)
		self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
		self.dropout1 = nn.Dropout2d(0.1)
		self.fc1 = nn.Linear(17472, 14000)
		self.fc2 = nn.Linear(14000, 10177)

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = self.conv3(x)
		x = F.relu(x)
		x = self.dropout1(x)
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.dropout1(x)
		x = self.fc2(x)
		output = F.log_softmax(x, dim=1)
		return output


def normal_init(m, mean, std):
	if isinstance(m, nn.Linear):
		m.weight.data.normal_(mean, std)
		m.bias.data.zero_()


def loss(desired_output, network_output):
	mse = torch.mean(((network_output - desired_output) ** 2))
	return mse
