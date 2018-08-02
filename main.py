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


def main(args):
  epochs = args.epochs
  batch_size = args.batch_size
  val_frac = args.val_frac
  lr = args.lr
  momentum = args.momentum
  total_it = args.total_it

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)

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

  net = Net()
  net.to(device)
  criterion = nn.MSELoss()
  optimizer = optim.SGD(net.parameters(), momentum=momentum, lr=lr)

  log_every = len(dataloader_train)//2
 
  n_it, lr_it = 0, 0
  while n_it < total_it: 
    train(dataloader_train, device, net, criterion, optimizer)
    test(dataloader_eval, device, net, criterion, n_it)
   
    n_it += len(dataloader_train)
    if lr_it >= 30000:
      d = optimizer.state_dict()
      d['param_groups'][0]['lr'] /= 10.0
      optimizer.load_state_dict(d)
      lr_it = 0


def train(dataloader_train, device, net, criterion, optimizer):
  net.train()
  for i, data in enumerate(dataloader_train):
    optimizer.zero_grad()
    inputs, labels = data['image'].to(device), data['label'].to(device)
   
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()


def test(dataloader_eval, device, net, criterion, n_it):
  net.eval()
  with torch.no_grad():

    running_loss = 0.0
    for data_eval in dataloader_eval:
      inputs, labels = data_eval['image'].to(device), data_eval['label'].to(device)
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      running_loss += loss.item()
    running_loss /= len(dataloader_eval)
  
    print('{:d} iter.:  {:0.4f}'.format(n_it, running_loss))
    #print('true labels')
    #print(labels.detach().cpu().numpy()[:3])
    #print('predicted labels')
    #print(outputs.detach().cpu().numpy()[:3])


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', type=int, default=2)
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--val_frac', type=float, default=0.05)
  parser.add_argument('--lr', type=float, default=0.005)
  parser.add_argument('--momentum', type=float, default=0.9)
  parser.add_argument('--total_it', type=int, default=90000)
  args = parser.parse_args(sys.argv[1:])
  main(args) 
