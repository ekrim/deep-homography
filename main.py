import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import Net
from pipeline import HomographyDataset 


def train(dataloader, epochs):
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)

  net = Net()
  net.to(device)
  criterion = nn.MSELoss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
 
  for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
      optimizer.zero_grad()
      inputs, labels = data['image'].to(device), data['label'].to(device)
   
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      if i % 2000 == 1999:
        print('epoch {:d}, batch {:d}: loss {:0.3f}'.format(epoch, i, running_loss/2000))
        running_loss = 0.0


if __name__ == '__main__':
  epochs, batch_size = 2, 2
  
  dataset = HomographyDataset()
  dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4)

  train(dataloader, epochs)
