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
               
class Down(nn.Module):
  def __init__(self, in_channels, out_channels, kernel, stride, padding):
    super().__init__()
    self.step = nn.Sequential(
        nn.Conv3d(in_channels= in_channels, out_channels= out_channels, kernel_size= kernel, stride=stride, padding=padding),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(True),
        )
  
  def forward(self, x):
    x =self.step(x)
    return x
    

class MyNet_Simple(nn.Module):
  def __init__(self, input_ch, n_channels, num_classes, printb):
    super().__init__()
    self.printb = printb
    self.down1 = Down(input_ch, n_channels, (3,3,3), (2,2,2), (1,1,1)) #64
    self.down2 = Down(n_channels, n_channels*2, (3,3,3), (2,2,2), (1,1,1)) #32
    self.down3 = Down(n_channels*2, n_channels*4, (3,3,3), (2,2,2), (1,1,1)) #16
    self.down4 = Down(n_channels*4, n_channels*8, (3,3,3), (2,2,2), (1,1,1)) #8
    self.down5 = Down(n_channels*8, n_channels*16, (3,3,3), (2,2,2), (1,1,1)) #4
    self.down6 = Down(n_channels*16, n_channels*32, (3,3,3), (2,2,2), (1,1,1)) #2
    
    self.avg = nn.AdaptiveAvgPool3d((1,1,1))
 
    
    self.end =  nn.Sequential(  
        nn.Linear(n_channels*32, num_classes),
        )

      
  def forward(self, x):
    if self.printb:
        print(x.shape)
    x = self.down1(x)   
    if self.printb:
        print(x.shape)
    x = self.down2(x) 
    if self.printb:
        print(x.shape)
    x = self.down3(x) 
    if self.printb:
        print(x.shape)
    x = self.down4(x)  
    if self.printb:
        print(x.shape)
    x = self.down5(x)  
    if self.printb:
        print(x.shape)
    x = self.down6(x) 
    if self.printb:
        print(x.shape)
    
    x = self.avg(x)
    if self.printb:
        print(x.shape)

    x = torch.flatten(x,1)
    if self.printb:
        print(x.shape)
    
    x = self.end(x)
    
    return x 
    
class Identity(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.step = nn.Sequential(
        nn.Conv3d(in_channels= in_channels, out_channels= in_channels, kernel_size= (3,3,3), stride=(1,1,1), padding=(1,1,1)),
        nn.BatchNorm3d(in_channels),
        nn.ReLU(True),
        nn.Conv3d(in_channels= in_channels, out_channels= in_channels, kernel_size= (3,3,3), stride=(1,1,1), padding=(1,1,1)),
        nn.BatchNorm3d(in_channels),
        )
    self.join = nn.ReLU(True)
  
  def forward(self, x):
    identity = x
    x = self.step(x)
    x += identity
    x = self.join(x)
    return x

class Dim_Red(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.step = nn.Sequential(
        nn.Conv3d(in_channels= in_channels, out_channels= out_channels, kernel_size= (1,1,1), stride=(2,2,2), padding=(0,0,0)),
        nn.BatchNorm3d(out_channels),
        )
  def forward(self, x):
    x =self.step(x)
    return x

class Branch(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.step = nn.Sequential(
        nn.Conv3d(in_channels= in_channels, out_channels= out_channels, kernel_size= (3,3,3), stride=(2,2,2), padding=(1,1,1)),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(True),
        nn.Conv3d(in_channels= out_channels, out_channels= out_channels, kernel_size= (3,3,3), stride=(1,1,1), padding=(1,1,1)),
        nn.BatchNorm3d(out_channels),
        )
  
  def forward(self, x):
    x = self.step(x)
    return x
    
class ReductionBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.step2_a = Dim_Red(in_channels, out_channels) #32
    self.step2_b =  Branch(in_channels, out_channels)#32
    self.join = nn.ReLU(True)
    
  def forward(self, x):
    x_a = self.step2_a(x)
    x_b = self.step2_b(x)
    x = x_a + x_b
    x = self.join(x)
    return x
    
    
class MyNet_ResNet(nn.Module):
  def __init__(self, input_ch, n_channels, num_classes, printb):
    super().__init__()
    self.printb = printb
    
    self.step1 = Down(input_ch, n_channels, (7,7,7), (1,1,1), (3,3,3))   #128
    self.maxPool = nn.MaxPool3d(kernel_size = (2,2,2)) #64
    self.identity1_1 = Identity(n_channels)
    
    #------------------> 1
    self.identity1 = Identity(n_channels)  #64
    #------------------> 2
    self.step2 = ReductionBlock(n_channels, n_channels*2) #32
    
    #------------------> 3
    self.identity2 = Identity(n_channels*2)  # 32
    #------------------> 4
    self.step3 = ReductionBlock(n_channels*2, n_channels*4) # 16
    
    #------------------> 5
    self.identity3 = Identity(n_channels*4)  # 16
    #------------------> 6
    self.step4 = ReductionBlock(n_channels*4, n_channels*8) # 8
    
    #------------------> 7
    self.identity4 = Identity(n_channels*8)  # 8
    #------------------> 8
    self.step5 = ReductionBlock(n_channels*8, n_channels*16) # 4
    
    #------------------> 9
    self.identity5 = Identity(n_channels*16)  # 4
    self.avg = nn.AdaptiveAvgPool3d((1,1,1))
    
    
    self.end =  nn.Sequential(  
        nn.Linear(n_channels*16,num_classes),
        )
        
  def forward(self, x):
    if self.printb:
      print(x.shape)
      
    x = self.step1(x)
    if self.printb:
      print(x.shape)
    x = self.maxPool(x)
    if self.printb:
      print(x.shape)
    x = self.identity1_1(x)
    if self.printb:
      print(x.shape)
    
    x = self.identity1(x)
    x = self.step2(x)
    if self.printb:
      print(x.shape)
      
    x = self.identity2(x)
    x = self.step3(x)
    if self.printb:
      print(x.shape)
      
    x = self.identity3(x)    
    x = self.step4(x)
    if self.printb:
      print(x.shape)
     
    x = self.identity4(x)     
    x = self.step5(x)  
    if self.printb:
      print(x.shape)
      
    assert x.shape[2] == 4
    x = self.identity5(x)
    
    x = self.avg(x)
    if self.printb:
        print(x.shape)
        

    x = torch.flatten(x,1)
    if self.printb:
        print(x.shape)
    
    x = self.end(x)
    
    return x