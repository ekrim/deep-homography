import sys
import argparse
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from models import Net
from pipeline import HomographyDataset 


def train(dataloader_train, dataloader_eval, epochs):
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)

  net = Net()
  net.to(device)
  criterion = nn.MSELoss()
  optimizer = optim.Adam(net.parameters(), lr=0.001)

  log_every = len(dataloader_train)//2
 
  for epoch in range(epochs):
    for i, data in enumerate(dataloader_train):
      optimizer.zero_grad()
      inputs, labels = data['image'].to(device), data['label'].to(device)
   
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
     
      if i % log_every == log_every-1:
        running_loss = 0.0
        for data_eval in dataloader_eval:
          inputs, labels = data_eval['image'].to(device), data_eval['label'].to(device)
          outputs = net(inputs)
          loss = criterion(outputs, labels)
          running_loss += loss.item()
        running_loss /= len(dataloader_eval)
        print('epoch {:d}, batch {:d}: loss {:0.3f}'.format(epoch, i, running_loss))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', type=int, default=2)
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--val_frac', type=float, default=0.05)
  args = parser.parse_args(sys.argv[1:])
  epochs = args.epochs
  batch_size = args.batch_size
  val_frac = args.val_frac
  
  dataset_train = HomographyDataset(val_frac=val_frac, mode='train')
  dataset_eval = HomographyDataset(val_frac=val_frac, mode='eval')
  dataloader_train= DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4)

  dataloader_eval = DataLoader(
    dataset_eval,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2)

  train(dataloader_train, dataloader_eval, epochs)
