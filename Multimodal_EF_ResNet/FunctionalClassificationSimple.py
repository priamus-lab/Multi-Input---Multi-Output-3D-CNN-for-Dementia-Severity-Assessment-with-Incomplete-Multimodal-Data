import torch
import scipy.io as sio
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
import time
import os
import matplotlib.pyplot as plt
from random import random, sample, shuffle, randrange
from scipy.io import loadmat
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.datasets import DatasetFolder
from torchvision.transforms import  ToTensor
import torch.nn.functional as F
import math 
from sklearn.metrics import balanced_accuracy_score
import torchio as tio
import resnet

#-----------------------------------> TRASFORMAZIONI
def rangeNormalizationIntra(x, supLim, infLim):
  #normalizzazione nel range
  x_norm = ( (x - np.min(x)) / (np.max(x)- np.min(x)) )*(supLim - infLim) + infLim
  assert np.min(x_norm) >= infLim
  assert np.max(x_norm) <= supLim
  return x_norm
  
def rangeNormalizationInter(x, max_p, min_p, supLim, infLim):
  x_norm = ( (x - min_p) / (max_p - min_p) )*(supLim - infLim) + infLim
  assert np.min(x_norm) >= infLim
  assert np.max(x_norm) <= supLim
  return x_norm
  
def np_imadjust(x, q1,q2):
  #applico un ehancement
  assert q1<q2
  assert q1+q2 == 1
  qq = np.quantile(x, [q1, q2])
  new = np.clip(x, qq[0], qq[1])
  return new
 
class ToTensor3D(torch.nn.Module):  
  def __init__(self):
    super().__init__()
  
  def forward(self, tensor):
    y_new = torch.from_numpy(tensor.transpose(3,2,0,1))
    return y_new

  def __repr__(self):
    return self.__class__.__name__ + '()'
  
class RandomZFlip(torch.nn.Module):
  def __init__(self, p=0.5):
    self.p = p                          
    super().__init__()                 

  def forward(self, img):

    if random() < self.p:
      img = torch.flip(img, [1])
    return img
    
  def __repr__(self):
    return self.__class__.__name__ + '(p={})'.format(self.p)

class NormalizeInRange(torch.nn.Module):
  def __init__(self, supLim, infLim, type_n, max_p, min_p):
    self.supLim = supLim
    self.infLim = infLim
    self.type_n = type_n
    self.max_p = max_p
    self.min_p = min_p
    super().__init__()
    
  def forward(self, img):
    if self.type_n == 'range_intra':
      x_norm = ( (img - torch.min(img)) / (torch.max(img)- torch.min(img)) )*(self.supLim - self.infLim) + self.infLim
      assert torch.min(x_norm) >= self.infLim
      assert torch.max(x_norm) <= self.supLim
      
    elif self.type_n == 'range_inter':
      x_norm = ( (img - self.min_p) / (self.max_p- self.min_p) )*(self.supLim - self.infLim) + self.infLim
      assert torch.min(x_norm) >= self.infLim
      assert torch.max(x_norm) <= self.supLim
    
    else: 
      x_norm = img
 
    return x_norm
    
  def __repr__(self):
    return self.__class__.__name__ + '(supLim={}, infLim = {}, type_n = {}, max_p = {}, min_p = {})'.format(self.supLim, self.infLim, self.type_n, self.max_p, self.min_p)
          
class Resize3D(torch.nn.Module):  
  def __init__(self, size=(32,32,32), enable_zoom = False):
    self.size = size    
    self.enable_zoom = enable_zoom    
    super().__init__()         

  def forward(self, tensor):
    if self.enable_zoom:
      img = F.interpolate( tensor.unsqueeze(0), self.size, align_corners =True, mode='trilinear').squeeze(0)
    else: 
      img = tensor
    return img
  
  def __repr__(self):
    return self.__class__.__name__ + '(size={}, enable_zoom = {})'.format(self.size, self.enable_zoom)
     
def getFilesForSubset(basepath, list_classes, include_patient):
  ListFiles=[]
  for c in list_classes:
    listofFiles = os.listdir(basepath + '/' + c)
    for file in listofFiles:
      if include_patient(basepath + '/' + c + '/' + file):
        ListFiles.append((basepath + '/' + c + '/' + file, list_classes.index(c)))
  return ListFiles

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  
#------------------------------------> RETE

class MyMedicalResNet(nn.Module):
  def __init__(self, typeResNet, weightPath, num_classes, inputChannel):
    super().__init__()
    if typeResNet == 'resnet50':
      self.backBone = resnet.resnet50(sample_input_D = 28, sample_input_H = 28, sample_input_W=28, num_seg_classes=num_classes)
    
    elif typeResNet == 'resnet34':
      self.backBone = resnet.resnet34(sample_input_D = 28, sample_input_H = 28, sample_input_W=28, num_seg_classes=num_classes)
    
    elif typeResNet == 'resnet10':
      self.backBone = resnet.resnet10(sample_input_D = 28, sample_input_H = 28, sample_input_W=28, num_seg_classes=num_classes)
    
    elif typeResNet == 'resnet18':
      self.backBone = resnet.resnet18(sample_input_D = 28, sample_input_H = 28, sample_input_W=28, num_seg_classes=num_classes)
      
    print('Loading from ' + weightPath)
    self.net_dict = self.backBone.state_dict()
    pretrain = torch.load(weightPath)
    pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in self.net_dict.keys()}
    self.net_dict.update(pretrain_dict)
    self.backBone.load_state_dict(self.net_dict)
    
    #cambia input
    self.backBone.conv1 = nn.Conv3d(in_channels= inputChannel, out_channels= 64, kernel_size= (7,7,7), stride=(2,2,2), padding=(3,3,3), bias=False)
    self.backBone.conv_seg = nn.AdaptiveAvgPool3d((1,1,1))
    self.end = nn.Sequential(
      nn.Linear(512,num_classes)
      
    )
    
  def forward(self, x):
    x = self.backBone(x)
    x = torch.flatten(x, 1)
    x = self.end(x)
    return x
   