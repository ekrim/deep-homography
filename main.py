import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(2, 16, 5)
    self.pool1 = nn.AvgPool2d(4)
    self.conv2 = nn.Conv2d(16, 32, 4)
    self.pool2 = nn.AvgPool2d(3)
    self.conv3 = nn.Conv2d(32, 1, 3)
    self.fc1 = nn.Linear(1*9*9, 64)
    self.fc2 = nn.Linear(64, 64)
    self.fc3 = nn.Linear(64, 64)
    self.fc4 = nn.Linear(64, 64)
    self.fc5 = nn.Linear(64, 8)
    
  def forward(self, x):
    x = self.pool1(F.relu(self.conv1(x)))
    x = self.pool2(F.relu(self.conv2(x)))
    x = F.relu(self.conv3(x))
    x = x.view(-1, 1*9*9)
    x = F.relu(self.fc1(x)) 
    x = F.relu(self.fc2(x)) 
    x = F.relu(self.fc3(x)) 
    x = F.relu(self.fc4(x)) 
    x = self.fc5(x) 
    return x


if __name__ == '__main__':
  net = Net()
