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


def read_and_convert(f, edges):
  big_img = cv2.imread(f)
  assert big_img is not None, 'no such file'
  big_img = big_img[edges[0]:edges[1], edges[2]:edges[3]]
  img = cv2.cvtColor(big_img, cv2.COLOR_BGR2GRAY)
  return big_img, img

  
def prep_batch(f1, f2, edges, n_patches, big_patch, patch_size):

  big_img1, img1 = read_and_convert(f1, edges)
  big_img2, img2 = read_and_convert(f2, edges)

  batch = np.zeros((n_patches, patch_size, patch_size, 2)).astype(np.float32) 
 
  row_center = img1.shape[0]//2, img1.shape[1]//2
  for i in range(n_patches):
     
    delta_row = np.random.randint(0, img1.shape[0] - big_patch)
    delta_col = np.random.randint(0, img1.shape[1] - big_patch)
    patch1 = img1[delta_row:delta_row+big_patch, delta_col:delta_col+big_patch]
    patch2 = img2[delta_row:delta_row+big_patch, delta_col:delta_col+big_patch]

    patch1 = cv2.resize(patch1, (patch_size, patch_size))
    patch2 = cv2.resize(patch2, (patch_size, patch_size))

    batch[i] = np.concatenate([patch1[:,:,None], patch2[:,:,None]], axis=2).astype(np.float32)

  return big_img1, big_img2, batch


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
   
    #h = np.linalg.inv(cv2.findHomography(pts1, pts2)[0])
    h = cv2.findHomography(pts1, pts2)[0]
    
    return h


if __name__ == '__main__':
  f1, f2 = 'house_1.jpg', 'house_3.jpg' 
  edges = [1200, 2600, 300, 3000]
  n_patches = 20
  patch_size = 128
  big_patch = 9*patch_size

  img1, img2, batch = prep_batch(f1, f2, edges, n_patches, big_patch, patch_size)
  h = pred_homography(batch, patch_size)
  new_img = cv2.warpPerspective(img1, h, (3000, 1400))
  plt.figure(1)
  plt.imshow(img1[:,:,[2,1,0]])
  plt.figure(2)
  plt.imshow(new_img[:,:,[2,1,0]])
  plt.figure(3)
  plt.imshow(img2[:,:,[2,1,0]])
  plt.show()
