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


class Normalize:
  def __call__(self, sample):
    image, label = sample['image'], sample['label']
    image = image.transpose((2,0,1))
    image = 2*(image/255) - 1
    #label = label/128

    return {'image': torch.from_numpy(image),
            'label': torch.from_numpy(label)}


class HomographyDataset(Dataset):
  def __init__(self, val_frac=0.05, mode='train'): 
    img_dir = 'data/synth_data'
    assert os.path.isdir(img_dir), 'Download the MSCOCO dataset and prepare it' 
    self.transforms = transforms.Compose([
      Normalize()])
  
    with open('data/label_file.txt', 'r') as f:
      num_and_label = [line.rstrip().rstrip(',').split(';') for line in f]

    L = len(num_and_label)
    idx = int(val_frac*L)
    
    if mode == 'train':
      self.num_and_label = num_and_label[idx:]
    elif mode == 'eval':
      self.num_and_label = num_and_label[:idx]
    else:
      raise ValueError('no such mode')
      
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
      'image': img,
      'label': label}
    sample = self.transforms(sample)
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
