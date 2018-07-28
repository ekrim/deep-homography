import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(2, 16, 5)
    self.pool1 = nn.AvgPool2d(4)
    self.bn1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16, 32, 4)
    self.pool2 = nn.AvgPool2d(3)
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 1, 3)
    self.fc1 = nn.Linear(1*7*7, 64)
    self.fc2 = nn.Linear(64, 64)
    self.fc5 = nn.Linear(64, 8)
    
  def forward(self, x):
    x = self.pool1(F.relu(self.conv1(x)))
    x = self.bn1(x)
    x = self.pool2(F.relu(self.conv2(x)))
    x = self.bn2(x)
    x = F.relu(self.conv3(x))
    x = x.view(-1, 1*7*7)
    x = F.relu(self.fc1(x)) 
    x = F.relu(self.fc2(x)) 
    x = F.tanh(self.fc5(x))
    return x


if __name__ == '__main__':
  x = np.random.randn(3,2,128,128).astype(np.float32)
  x = torch.from_numpy(x)
  net = Net()
  net.forward(x)
