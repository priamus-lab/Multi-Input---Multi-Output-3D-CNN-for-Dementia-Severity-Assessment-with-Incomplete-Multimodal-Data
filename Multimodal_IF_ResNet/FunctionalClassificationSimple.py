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
  def __init__(self, typeResNet, weightPath, num_classes):
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
    
    self.backBone.conv_seg = nn.AdaptiveAvgPool3d((1,1,1))
    self.end = nn.Sequential(
      nn.Linear(512,num_classes)
      
    )
    
  def forward(self, x):
    x = self.backBone(x)
    x = torch.flatten(x, 1)
    x = self.end(x)
    return x
    
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
    

class EmptyLayer(nn.Module):
  def __init__(self):
    super(EmptyLayer, self).__init__()
  def forward(self, x):
    return x
    
class MyNet_Multimodal(nn.Module):
  def __init__(self, input_ch, n_channels, num_classes, model_type, path_mri, path_pet, printb):
    super(MyNet_Multimodal, self).__init__()
    self.model_type = model_type
    self.path_mri = path_mri
    self.printb = printb
    
    if self.model_type == 'standard':
      #input_ch, n_channels, num_classes, printb
      self.mri_part = MyNet_Simple(1,n_channels,num_classes, False)
      self.pet_part = MyNet_Simple(1,n_channels,num_classes, False)
    else:
      self.mri_part = MyNet_ResNet(1,n_channels,num_classes, False)
      self.pet_part = MyNet_ResNet(1,n_channels,num_classes, False)
      
    print('Loading mri from ' + path_mri)
    print('Loading pet from ' + path_pet)   
    self.mri_part.load_state_dict(torch.load(path_mri))
    self.pet_part.load_state_dict(torch.load(path_pet))
    
    self.fc_mri = self.mri_part.end
    self.fc_pet = self.pet_part.end
    
    self.mri_part.end = EmptyLayer()
    self.pet_part.end = EmptyLayer()
    
    self.fc_both =  nn.Sequential(  
      nn.Linear(n_channels*64*2,n_channels*64),
      nn.ReLU(inplace= True),
      nn.Linear(n_channels*64,n_channels*32),
      nn.ReLU(inplace= True),
      nn.Linear(n_channels*32,num_classes)
    )
    
  def forward(self, x, mode):
    if mode == 'mri':
      x = self.mri_part(x)
      x = self.fc_mri(x)
    
    if mode == 'pet':
      x = self.pet_part(x)
      x = self.fc_pet(x)
    
    if mode == 'multi':
      m_x = self.mri_part(x[0])
      p_x = self.pet_part(x[1])
      
      x = torch.cat((m_x,p_x), 1)
      
      x = self.fc_both(x)
      
    return x
 
class MyNet_MultimodalMedicalNet(nn.Module):
  def __init__(self, input_ch, num_classes, model_type, weightPath, path_mri, path_pet, printb):
    super(MyNet_MultimodalMedicalNet, self).__init__()
    self.model_type = model_type
    self.path_mri = path_mri
    self.printb = printb
    

    self.mri_part = MyMedicalResNet(model_type, weightPath, num_classes)
    self.pet_part = MyMedicalResNet(model_type, weightPath, num_classes)
      
    print('Loading mri from ' + path_mri)
    print('Loading pet from ' + path_pet)   
    self.mri_part.load_state_dict(torch.load(path_mri, map_location='cuda:1'))
    self.pet_part.load_state_dict(torch.load(path_pet, map_location='cuda:0' ))
    
    
    self.fc_mri = self.mri_part.end
    self.fc_pet = self.pet_part.end
    
    self.mri_part.end = EmptyLayer()
    self.pet_part.end = EmptyLayer()
    
    self.fc_both =  nn.Sequential(  
      nn.Linear(512*2,512),
      nn.ReLU(inplace= True),
      nn.Linear(512,num_classes),
    )
    
    self.mri_part.to('cuda:1')
    self.pet_part.to('cuda:0')
    
    self.fc_mri.to('cuda:0')
    self.fc_pet.to('cuda:0')
    
    self.fc_both.to('cuda:0')
    
  def forward(self, x, mode):
    if mode == 'mri':
      x = x.to('cuda:1')
      x = self.mri_part(x)
      x = x.to('cuda:0')
      x = self.fc_mri(x)
    
    if mode == 'pet':
      x = x.to('cuda:0')
      x = self.pet_part(x)
      x = self.fc_pet(x)
    
    if mode == 'multi':
      x0 = x[0].to('cuda:1')
      x1 = x[1].to('cuda:0')
      
      m_x = self.mri_part(x0)
      p_x = self.pet_part(x1)
      
      m_x = m_x.to('cuda:0')
      x = torch.cat((m_x,p_x), 1)
      
      x = self.fc_both(x)
      
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
    
    self.step1 = Down(input_ch, n_channels, (7,7,7), (2,2,2), (3,3,3))   #128
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
    #------------------> 10
    #self.step6 = ReductionBlock(n_channels*16, n_channels*32) # 2
    #------------------> 11
    #self.identity6 = Identity(n_channels*32)  # 2
    
    
    self.avg = nn.AdaptiveAvgPool3d((1,1,1))
    
    
    self.end =  nn.Sequential(  
        nn.Linear(n_channels*16,n_channels*4),
        nn.ReLU(inplace= True),
        nn.Linear(n_channels*4,num_classes),
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
      
    x = self.identity5(x)
    
    
    x = self.avg(x)
    if self.printb:
        print(x.shape)
        

    x = torch.flatten(x,1)
    if self.printb:
        print(x.shape)
    
    x = self.end(x)
    
    return x

