import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models import Net


def normalize_batch(image_batch):
  image_batch = image_batch.transpose((0,3,1,2))
  image_batch = 2*(image_batch/255) - 1
  return torch.from_numpy(image_batch)


def read_and_convert(f):
  img = cv2.imread(f)
  assert img is not None, 'no such file'
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return img

  
def prep_batch(f1, f2, patch_size):

  img1 = read_and_convert(f1)
  img2 = read_and_convert(f2)

  patch1 = cv2.resize(img1, (patch_size, patch_size))
  patch2 = cv2.resize(img2, (patch_size, patch_size))

  batch = np.concatenate([patch1[:,:,None], patch2[:,:,None]], axis=2).astype(np.float32)[None, :, :, :]

  return img1, img2, batch


def pred_homography(batch, patch_size, model_file='homography_model.pytorch', scale=32):
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 
  batch = normalize_batch(batch).to(device)
  net = Net()
  net.load_state_dict(torch.load(model_file))  
  net.eval().to(device)
  with torch.no_grad():
    output = net(batch).detach().cpu().numpy()*scale
    mean_shift = np.mean(output, axis=0)
 
    pts1 = np.float32([0, 0, patch_size, 0, patch_size, patch_size, 0, patch_size]).reshape(-1,1,2)
    pts2 = mean_shift.reshape(-1,1,2) + pts1
   
    h = np.linalg.inv(cv2.findHomography(pts1, pts2)[0])
    
    return h


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--patch_size', default=128, type=int, help='image patch size to resize to')
  parser.add_argument('--i', default=-1, type=int, help='number of image pair in training set')
  parser.add_argument('--file1', default='', type=str, help='file of image 1')  
  parser.add_argument('--file2', default='', type=str, help='file of image 2')  
  args = parser.parse_args(sys.argv[1:])
  patch_size = args.patch_size
  if args.i > -1: 
    i = args.i
    f1, f2 = 'data/synth_data/{:09d}_orig.jpg'.format(i), 'data/synth_data/{:09d}_warp.jpg'.format(i)

  elif args.file1 != '' and args.file1 != '':
    f1, f2 = args.file1, args.file2

  else:
    raise ValueError('specify file name pair or integer for image pair in training data')

  img1, img2, batch = prep_batch(f1, f2, patch_size)
  h = pred_homography(batch, patch_size)
  new_img = cv2.warpPerspective(img1, h, (patch_size, patch_size))
  plt.figure(1)
  plt.subplot(131)
  plt.imshow(img1, cmap='Greys_r')
  plt.title('original img')
  plt.subplot(133)
  plt.imshow(new_img, cmap='Greys_r')
  plt.title('orig img warped \n by pred homography')
  plt.subplot(132)
  plt.imshow(img2, cmap='Greys_r')
  plt.title('"warped" img')
  plt.show()
