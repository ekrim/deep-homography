import struct
import os
import urllib
import zipfile
import pickle
import PIL
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


def jpg_reader(filename):
  img = PIL.Image.open(filename)
  return np.asarray(img)


class HomographyDataset(Dataset):
  def __init__(self): 
    img_dir = 'data/synth_data'
    assert os.path.isdir(img_dir), 'Download the MSCOCO dataset and prepare it' 
  
    with open('data/label_file.txt', 'r') as f:
      self.num_and_label = [line.rstrip().rstrip(',').split(';') for line in f]
    
  def __len__(self):
    return len(self.num_and_label)

  def __getitem__(self, idx):
    num = self.num_and_label[idx][0]
    input_file_orig = 'data/synth_data/{:s}_orig.jpg'.format(num)
    input_file_warp = 'data/synth_data/{:s}_warp.jpg'.format(num)
    img_orig = jpg_reader(input_file_orig)
    img_warp = jpg_reader(input_file_warp)
    img = np.concatenate([img_orig[:,:,None], img_warp[:,:,None]], axis=2).astype(np.float32)
   
    label_str = self.num_and_label[idx][1] 
    label = np.array([float(el) for el in label_str.split(',')]).astype(np.float32)
    sample = {
      'image': torch.from_numpy(img.transpose((2,0,1))),
      'label': torch.from_numpy(label)}
    return sample


if __name__ == '__main__':
  dataset = HomographyDataset()
  dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4)

  for i_batch, sample_batch in enumerate(dataloader):
    print(sample_batch['image'].size())
    print(sample_batch['label'].size())
