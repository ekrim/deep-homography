import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.sample = nn.AvgPool2d(2)
    self.conv1 = nn.Conv2d(2, 64, 3, padding=1)
    self.bn1 = nn.BatchNorm2d(64)
    self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
    self.pool1 = nn.MaxPool2d(2)
    self.bn2 = nn.BatchNorm2d(64)
    self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
    self.bn3 = nn.BatchNorm2d(64)
    self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
    self.pool2 = nn.MaxPool2d(2)
    self.bn4 = nn.BatchNorm2d(64)
    self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
    self.bn5 = nn.BatchNorm2d(128)
    self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
    self.pool3 = nn.MaxPool2d(2)
    self.bn6 = nn.BatchNorm2d(128)
    self.conv7 = nn.Conv2d(128, 128, 3, padding=1)
    self.bn7 = nn.BatchNorm2d(128)
    self.conv8 = nn.Conv2d(128, 128, 3, padding=1)
    self.drop1 = nn.Dropout(p=0.5)
    self.bn8 = nn.BatchNorm1d(16*16*128)
    self.fc1 = nn.Linear(16*16*128, 1024)
    self.drop2 = nn.Dropout(p=0.5)
    self.bn9 = nn.BatchNorm1d(1024)
    self.fc2 = nn.Linear(1024, 8)
    
  def forward(self, x):
    x = self.sample(x)
    x = self.bn1(F.relu(self.conv1(x)))
    x = self.bn2(self.pool1(F.relu(self.conv2(x))))
    x = self.bn3(F.relu(self.conv3(x)))
    x = self.bn4(self.pool2(F.relu(self.conv4(x))))
    x = self.bn5(F.relu(self.conv5(x)))
    x = self.bn6(self.pool3(F.relu(self.conv6(x))))
    x = self.bn7(F.relu(self.conv7(x)))
    x = F.relu(self.conv8(x))
    x = x.view(-1, 16*16*128)
    x = self.bn8(self.drop1(x))
    x = F.relu(self.fc1(x))
    x = self.bn9(self.drop2(x))
    x = self.fc2(x)
   
    return x


if __name__ == '__main__':
  x = np.random.randn(3,2,256,256).astype(np.float32)
  x = torch.from_numpy(x)
  net = Net()
  net.forward(x)
