from FunctionalClassificationSimple import*

def print_number_files(val_group):
  #{'MRI_CN'}    {'MRI_MILD'}    {'MRI_SEVERE'}    {'PET_CN'}    {'PET_MILD'}    {'PET_SEVERE'}    {'PAIRED_CN'}    {'PAIRED_MILD'}    {'PAIRED_SEVERE'}
  print('MRI_CN: ' + str(np.sum(val_group.MRI_CN.values)) +
        ' MRI_MILD: ' + str(np.sum(val_group.MRI_MILD.values)) + 
        ' MRI_SEVERE: ' + str(np.sum(val_group.MRI_SEVERE.values)) +
        ' PET_CN: ' + str(np.sum(val_group.PET_CN.values)) +        
        ' PET_MILD: ' + str(np.sum(val_group.PET_MILD.values)) + 
        ' PET_SEVERE: ' + str(np.sum(val_group.PET_SEVERE.values)) + 
        ' PAIRED_CN: ' + str(np.sum(val_group.PAIRED_CN.values)) +
        ' PAIRED_MILD: ' + str(np.sum(val_group.PAIRED_MILD.values)) + 
        ' PAIRED_SEVERE: ' + str(np.sum(val_group.PAIRED_SEVERE.values)))
        
        
#-----------------------------------> DATALOADER
def compute_crop(x_coord, y_coord, z_coord):  
  new_x_cord = [x_coord[0] -1, x_coord[1] -1]
  new_y_cord = [y_coord[0] -1, y_coord[1] -1]
  new_z_cord = [z_coord[0] -1, z_coord[1] -1] 
  
  cx = (new_x_cord[0] +  new_x_cord[1])/2
  cy = (new_y_cord[0] +  new_y_cord[1])/2
  cz = (new_z_cord[0] +  new_z_cord[1])/2
  
  raggio = np.max([(new_x_cord[1] -  new_x_cord[0] )/2, (new_y_cord[1] -  new_y_cord[0])/2, (new_z_cord[1] -  new_z_cord[0])/2])
  
  ffx = [cx-raggio, cx+raggio]
  ffy = [cy-raggio, cy+raggio]
  ffz = [cz-raggio, cz+raggio]

  ffx, range_x = check_coord(ffx, 0, 255)
  ffy, range_y = check_coord(ffy, 0, 255)
  ffz, range_z = check_coord(ffz, 0, 255)
  
  ffx = [int(ffx[0]), int(ffx[1])]
  ffy = [int(ffy[0]), int(ffy[1])]
  ffz = [int(ffz[0]), int(ffz[1])]

  assert range_x == range_y and range_y == range_z
  
  return ffx, ffy, ffz
  

def check_coord(xx, min, max):
    rangein = xx[1] - xx[0] + 1
    
    if xx[0]< min:
        xx[1]= xx[1] + (min - xx[0])
        xx[0] = 0
    
    if xx[1] > max:
        xx[0] = xx[0] - (xx[1] - max)
        xx[1] = max
        
    range_2 = xx[1] - xx[0] + 1
    assert range_2 == rangein 
    
    return xx,range_2
    
def readVolume_mri(path):
  x = loadmat(path)
  y= x['norm'][0][0][0].astype(np.float32)
  y = y[:,:,:, np.newaxis]
  return y
  
def readVolume_mri_zoom(path):
  x = loadmat(path)
  y= x['norm'][0][0][0].astype(np.float32)
  x_coord = x['norm'][0][0][1][0]
  y_coord = x['norm'][0][0][2][0]
  z_coord = x['norm'][0][0][3][0]
  
  ffx, ffy, ffz = compute_crop(x_coord,y_coord, z_coord)

  y = y[ffx[0]:(ffx[1]+1),ffy[0]:(ffy[1]+1),ffz[0]:(ffz[1]+1)]
  y = y[:,:,:, np.newaxis]
  return y
  
def readVolume_pet(path):
  x = loadmat(path)
  y= x['pet'][0][0][0].astype(np.float32)
  y = y.transpose(1,0,2)
  y = y[:,:,:, np.newaxis]
  return y
  
def readVolume_pet_zoom(path):
  x = loadmat(path)
  y= x['pet'][0][0][0].astype(np.float32)
  
  x_coord = x['pet'][0][0][1][0]
  y_coord = x['pet'][0][0][2][0]
  z_coord = x['pet'][0][0][3][0]
  
  ffx, ffy, ffz = compute_crop(x_coord,y_coord, z_coord)
  
  y = y[ffx[0]:(ffx[1]+1),ffy[0]:(ffy[1]+1),ffz[0]:(ffz[1]+1)]
  y = y.transpose(1,0,2)
  
  y = y[:,:,:, np.newaxis]
  return y
  

def np_computeMeanAndStd_all(train_files, readVolume, type_n, min_p, max_p, infLim, supLim): 
  channel_sum, channel_sqared_sum = 0.0,0.0
  num_batches = 0
  
  for f,l in train_files:
    data = readVolume(f)
    
    if type_n == 'range_intra':
      #(x, supLim, infLim)
      data = rangeNormalizationIntra(data, supLim, infLim)
    elif type_n == 'range_inter':
      #(x, max_p, min_p, supLim, infLim)
      data = rangeNormalizationInter(data, max_p, min_p, supLim, infLim)
      
    ss = np.array(data.shape)
    channel_sum += np.sum(data)
    channel_sqared_sum  += np.sum(data**2)
    num_batches += ss.prod()  #devo fare il prodotto delle dimensioni
  
  mean =channel_sum/num_batches
  std = (channel_sqared_sum/num_batches - mean**2)**0.5
  return mean, std
  
  
def np_computeMinMax(train_files, readVolume):   
  i = 0
  
  data = readVolume(train_files[i][0])
  min_p = np.min(data)
  max_p = np.max(data)
  
  for i in range(1,len(train_files)):
    data = readVolume(train_files[i][0])
    
    if np.min(data) < min_p:
      min_p = np.min(data)
      
    if np.max(data) > max_p:
      max_p = np.max(data)
      
  return min_p, max_p
  
  
class My_DatasetFolder(Dataset):
  def __init__(self, root,  transform, is_valid_file, list_classes, readVolume):
    self.root = root 
    self.transform = transform
    self.is_valid_file = is_valid_file
    self.list_classes = list_classes
    self.samples = self.__get_samples()
    self.readVolume = readVolume

  def __len__(self):
    return len(self.samples)

  def __get_samples(self):
    ListFiles=[]
    for c in self.list_classes:
      listofFiles = os.listdir(self.root + '/' + c)
      for file in listofFiles:
        if self.is_valid_file(self.root + '/' + c + '/' + file): 
          ListFiles.append((self.root + '/' + c + '/' + file, self.list_classes.index(c)))     
    return ListFiles

  def __getitem__(self, index: int):
    path, target = self.samples[index]
    volume = self.readVolume(path)
    if self.transform is not None:
      volume = self.transform(volume)
    return volume, target
    

class My_DatasetFolderMultiModal(Dataset):
  def __init__(self, root,  transform_mri, transform_pet, is_valid_file, list_classes, readerMRI, readerPET):
    self.root = root 
    self.transform_mri = transform_mri
    self.transform_pet = transform_pet
    self.is_valid_file = is_valid_file
    self.list_classes = list_classes
    self.samples = self.__get_samples()
    self.readerMRI = readerMRI
    self.readerPET = readerPET

  def __len__(self):
    return len(self.samples)

  def __get_samples(self):
    ListFiles=[]
    for c in self.list_classes:
      listofFiles = os.listdir(self.root + '/' + c)
      for file in listofFiles:
        if self.is_valid_file(self.root + '/' + c + '/' + file): 
          ListFiles.append((self.root + '/' + c + '/' + file, self.list_classes.index(c)))     
    return ListFiles

  def __getitem__(self, index: int):
    path, target = self.samples[index]
    volume_mri = self.readerMRI(path)
    volume_pet = self.readerPET(path)
    if self.transform_mri is not None:
      volume_mri = self.transform_mri(volume_mri)
    if self.transform_pet is not None:
      volume_pet = self.transform_pet(volume_pet)
    return volume_mri, volume_pet, target
    
class BalanceConcatDataset(ConcatDataset):  
  def __init__(self, datasets):
    l = max([len(dataset) for dataset in datasets])
    new_l = l
    
    for dataset in datasets: 
      old_samples = dataset.samples 
      while len(dataset) < new_l:
        #sample is without replacement
        dataset.samples += sample(old_samples, min(len(dataset), new_l - len(dataset))) #change 
        
    super(BalanceConcatDataset, self).__init__(datasets)