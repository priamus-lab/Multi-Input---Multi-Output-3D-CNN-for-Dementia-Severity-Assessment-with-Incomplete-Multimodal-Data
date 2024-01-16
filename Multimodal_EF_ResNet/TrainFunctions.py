from Utils_volume import*
from mpl_toolkits.axes_grid1 import ImageGrid

def train_loop_validation(model_conv, 
                          trainset, Val, test, 
                          start, num_epoch, 
                          loader_opts,
                          criterionCNN, optimizer_conv, 
                          best_acc, best_acc_m, best_acc_w, best_loss, best_epoca,
                          outputPath, device):
  
  for epochs in range(start, num_epoch + 1):
    
    TrainLoader = DataLoader(trainset, shuffle=True, **loader_opts)
    
    
    modelLoss_train = 0.0
    modelAcc_train = 0.0
    totalSize = 0

    model_conv.train() 
    totPred = torch.empty(0)
    totLabels = torch.empty(0)

    #-----------------------------------------------------------------------------> TRAIN      
    for inputs, labels in TrainLoader:
      
      inputs = inputs.type(torch.FloatTensor).to(device=device)
      labels = labels.to(device=device)
      
      optimizer_conv.zero_grad()
      model_conv.zero_grad()
       
      y = model_conv(inputs)
      outp, preds = torch.max(y, 1)   
      lossCNN = criterionCNN(y, labels) #media per batch
       
      lossCNN.backward()
      optimizer_conv.step()
       
      totPred = torch.cat((totPred, preds.cpu()))
      totLabels = torch.cat((totLabels, labels.cpu()))
       
      modelLoss_train += lossCNN.item() * inputs.size(0)
      totalSize += inputs.size(0)
      modelAcc_train += torch.sum(preds == labels.data).item()
      
      inputs = inputs.detach()
      del inputs
      labels = labels.detach()
      del labels
      y = y.detach()
      del y
      
    
    modelLoss_epoch_train = modelLoss_train/totalSize
    modelAcc_epoch_train  = modelAcc_train/totalSize
    
    totPred = totPred.numpy()
    totLabels = totLabels.numpy()
    acc = np.sum((totPred == totLabels).astype(int))/totalSize
    
    x = totLabels[np.where(totLabels == 0)]
    y = totPred[np.where(totLabels == 0)]
    acc_0_T = np.sum((x == y).astype(int))/y.shape[0]
    
    x = totLabels[np.where(totLabels == 1)]
    y = totPred[np.where(totLabels == 1)]
    acc_1_T = np.sum((x == y).astype(int))/x.shape[0]
    
    x = totLabels[np.where(totLabels == 2)]
    y = totPred[np.where(totLabels == 2)]
    acc_2_T = np.sum((x == y).astype(int))/x.shape[0]
    
    with open(outputPath + 'lossTrain.txt', "a") as file_object:
      file_object.write(str(modelLoss_epoch_train) +'\n')
    with open(outputPath + 'AccTrain.txt', "a") as file_object:
      file_object.write(str(modelAcc_epoch_train)+'\n')
      
    torch.save(model_conv.state_dict(), outputPath + 'train_weights.pth')
    print('[Epoch %d][TRAIN on %d [Loss: %.4f - ACC_T: %.4f - ACC_0: %.4f - ACC_1: %.4f - ACC_2: %.4f]]' %(epochs, totalSize, modelLoss_epoch_train, modelAcc_epoch_train, acc_0_T, acc_1_T, acc_2_T))
    #-----------------------------------------------------------------------------> VALIDATION 
    model_conv.eval()    
    modelAcc_epoch_val, weight_mean_val, mean_accuracy_val, modelLoss_epoch_val = ValidationStep(Val, loader_opts, model_conv, criterionCNN, device, outputPath)
   
    if epochs == 1 or (mean_accuracy_val > best_acc_m) or (weight_mean_val > best_acc_w) or ((mean_accuracy_val == best_acc_m) and (weight_mean_val == best_acc_w) and (modelLoss_epoch_val <= best_loss)):
      print('     .... Saving best weights ....')
      best_acc = modelAcc_epoch_val
      best_acc_w =  weight_mean_val
      best_acc_m = mean_accuracy_val
      best_loss = modelLoss_epoch_val
      best_epoca = epochs
                                                 
      #salvataggio dei migliori pesi sul validation
      torch.save(model_conv.state_dict(), outputPath + 'best_model_weights.pth')
      validateOnTest(test, loader_opts, model_conv, criterionCNN, device)      
    
    sio.savemat(outputPath + 'check_point.mat', {'best_acc': best_acc, 
                                                 'best_acc_m':best_acc_m,
                                                 'best_acc_w': best_acc_w,
                                                 'best_loss': best_loss,
                                                 'best_epoca': best_epoca,
                                                 'last_epoch': epochs})
  return model_conv
  
def ValidationStep(Val, loader_opts, model_conv, criterionCNN, device, outputPath):
  totalSize_val = 0
  modelLoss_val = 0.0
  modelAcc_val = 0.0

  totPred_val = torch.empty(0)
  totLabels_val = torch.empty(0)
                    
  ValLoader = DataLoader(Val, shuffle=True, **loader_opts)
  for inputs, labels in ValLoader:
    inputs = inputs.type(torch.FloatTensor).to(device=device)
    labels = labels.to(device=device)
    
    with torch.set_grad_enabled(False):
      y = model_conv(inputs)
      outp, preds = torch.max(y, 1)
      lossCNN = criterionCNN(y, labels)

    totPred_val = torch.cat((totPred_val, preds.cpu()))
    totLabels_val = torch.cat((totLabels_val, labels.cpu()))

    modelLoss_val += lossCNN.item() * inputs.size(0)  #Non pesata -> semplice media
    totalSize_val += inputs.size(0)
    modelAcc_val += torch.sum(preds == labels.data).item()
      
    inputs = inputs.detach()
    del inputs
    labels = labels.detach()
    del labels
    y = y.detach()
    del y
    
        
  modelLoss_epoch_val = modelLoss_val/totalSize_val
  modelAcc_epoch_val = modelAcc_val/totalSize_val

  totPred_val = totPred_val.numpy()
  totLabels_val = totLabels_val.numpy()
    
  x = totLabels_val[np.where(totLabels_val == 1)]
  y = totPred_val[np.where(totLabels_val == 1)]
  acc_1_V = np.sum((x == y).astype(int))/x.shape[0]
  n1 = x.shape[0]
    
  x = totLabels_val[np.where(totLabels_val == 0)]
  y = totPred_val[np.where(totLabels_val == 0)]
  acc_0_v = np.sum((x == y).astype(int))/x.shape[0]
  
  x = totLabels_val[np.where(totLabels_val == 2)]
  y = totPred_val[np.where(totLabels_val == 2)]
  acc_2_v = np.sum((x == y).astype(int))/x.shape[0]
 
  mean_accuracy_val = (acc_1_V + acc_0_v + acc_2_v)/3
  weight_mean_val =  balanced_accuracy_score(totLabels_val, totPred_val)
    
      
  with open(outputPath + 'lossVal.txt', "a") as file_object:
    file_object.write(str(modelLoss_epoch_val) +'\n')
    
  with open(outputPath + 'AccVal.txt', "a") as file_object:
    file_object.write(str(modelAcc_epoch_val)+'\n')
    
  with open(outputPath + 'AccVal_0.txt', "a") as file_object:
    file_object.write(str(acc_0_v)+'\n')
      
  with open(outputPath + 'AccVal_1.txt', "a") as file_object:
    file_object.write(str(acc_1_V)+'\n')
  
  with open(outputPath + 'AccVal_2.txt', "a") as file_object:
    file_object.write(str(acc_2_v)+'\n')
  
  print('    [VAL on %d [Loss: %.4f - ACC_T: %.4f - ACC_0: %.4f - ACC_1: %.4f - ACC_2: %.4f - ACC_Mean: %.4f - ACC_weig: %.4f]]' 
    %(totalSize_val, modelLoss_epoch_val, modelAcc_epoch_val, acc_0_v, acc_1_V, acc_2_v, mean_accuracy_val, weight_mean_val))
          
  return modelAcc_epoch_val, weight_mean_val, mean_accuracy_val, modelLoss_epoch_val
          
def validateOnTest(test, loader_opts, model_conv,criterionCNN, device):
  tot_size_test = 0
  model_loss_test = 0.0
  modelAcc_acc_test = 0.0
  totPred_test = torch.empty(0)
  totLabels_test = torch.empty(0)  

  TestLoader = DataLoader(test, shuffle=True, **loader_opts)
  
  for  inputs, labels in TestLoader:
    inputs = inputs.type(torch.FloatTensor).to(device=device)
    labels = labels.to(device=device)
    
    with torch.set_grad_enabled(False):    
      y = model_conv(inputs)
      outp, preds = torch.max(y, 1)
      lossCNN = criterionCNN(y, labels)

    totPred_test = torch.cat((totPred_test, preds.cpu()))
    totLabels_test = torch.cat((totLabels_test, labels.cpu()))

    model_loss_test += lossCNN.item() * inputs.size(0)  #Non pesata -> semplice media
    tot_size_test += inputs.size(0)
    modelAcc_acc_test += torch.sum(preds == labels.data).item()
    inputs = inputs.detach()
    del inputs
    labels = labels.detach()
    del labels
    y = y.detach()
    del y
        
  modelLoss_epoch_test = model_loss_test/tot_size_test
  modelAcc_epoch_test = modelAcc_acc_test/tot_size_test
      
  totPred_test = totPred_test.numpy()
  totLabels_test = totLabels_test.numpy()
      
  x = totLabels_test[np.where(totLabels_test == 1)]
  y = totPred_test[np.where(totLabels_test == 1)]
  acc_1_test = np.sum((x == y).astype(int))/x.shape[0]
      
  x = totLabels_test[np.where(totLabels_test == 0)]
  y = totPred_test[np.where(totLabels_test == 0)]
  acc_0_test = np.sum((x == y).astype(int))/y.shape[0] 
  
  x = totLabels_test[np.where(totLabels_test == 2)]
  y = totPred_test[np.where(totLabels_test == 2)]
  acc_2_test = np.sum((x == y).astype(int))/y.shape[0]
      
      

  print('        [TEST on %d [Loss: %.4f - ACC_T: %.4f - ACC_0: %.4f - ACC_1: %.4f - ACC_1: %.4f]]' 
        %(tot_size_test, modelLoss_epoch_test, modelAcc_epoch_test, acc_0_test, acc_1_test, acc_2_test))
  
## #-------------------------------------------------------------------------------> Predict function
def prediction_on_Test(model_conv, test, device):
  
  func = nn.Softmax(dim=1)
  predicted = pd.DataFrame()
  testFiles = test.samples
  
  for xx in range(0, len(testFiles)):
    path = testFiles[xx][0]
    inputs, label_true = test.__getitem__(xx)
    
    inputs = inputs.type(torch.FloatTensor).unsqueeze(0).to(device = device)
    parti = path.split('/')[-1].split('_')
    classe_string = path.split('/')[-2]
    
    #paz_mri_dmmm_pib_ddd.mat
    if len(parti)>3:
      app = parti[2]
      split_app = app.split('d')
      
      dMRI = int(split_app[1])
      
      app = parti[4].split('.')[0]
      split_app = app.split('d')
      dPET = int(split_app[1])
    else:
      dPET = 0
      app = parti[2].split('.')[0]
      split_app = app.split('d')
      dMRI = int(split_app[1])
      
    
    
    y = model_conv(inputs).detach().cpu()
    outp, preds = torch.max(y, 1)
    y = func(y) 
   

    
    for i in range(0, inputs.shape[0]):
      predicted = predicted.append({'filename': path.split('/')[-1],
                                    'Patient': parti[0],
                                    'dMRI':  dMRI,
                                    'dPET': dPET,
                                    'prob0': y[i,0].item(),
                                    'prob1': y[i,1].item(),
                                    'prob2': y[i,2].item(),
                                    'predicted': preds[i].item(),
                                    'true_class': label_true,
                                    'classe_string':  classe_string
                                    }, ignore_index=True)
    
    inputs = inputs.cpu()
    del inputs
    
  return predicted
  

def DefineRangeNormalization(normalizzazione_range, supLim, infLim, train_files_to_use, loader):
  min_p, max_p = 0.0, 0.0
  
  if normalizzazione_range == 'range_intra':
    #supLim, infLim, type_n, max_p, min_p
    norm_range = NormalizeInRange(supLim, infLim, normalizzazione_range, 0, 0)
  elif normalizzazione_range == 'range_inter':
    print('Computing max and min')
    min_p, max_p = np_computeMinMax(train_files_to_use, loader)
    norm_range = NormalizeInRange(supLim, infLim, normalizzazione_range, max_p, min_p)
  else:
    norm_range = NormalizeInRange(0, 0, normalizzazione_range, 0, 0)
  
  return norm_range, min_p, max_p
  
def Main_train(vali_set, test_set, patient_to_include, listClasses, newClasses, basePath_mri, basePath_pet, basePath_Paired,batchSize,
               outputPath, num_epoch, learningRate, weightDecay, continue_learning, model_type, weightPath, resize,  
               normalizzazione_range_mri, normalizzazione_range_pet,
               supLim_mri, infLim_mri, supLim_pet, infLim_pet, zscore_mri, zscore_pet, enable_zoom, enhance,
               device):


  include_train_patient = lambda path: ((path.split('/')[-1].split('_')[0] not in vali_set + test_set) and (path.split('/')[-1].split('_')[0] in patient_to_include))                        
  include_val_patient =  lambda path: ((path.split('/')[-1].split('_')[0] in vali_set) and (path.split('/')[-1].split('_')[0] in patient_to_include)) 
  include_test_patient =  lambda path: ((path.split('/')[-1].split('_')[0] in test_set) and (path.split('/')[-1].split('_')[0] in patient_to_include))
  
  train_files_to_use_mri = getFilesForSubset(basePath_mri, listClasses, include_train_patient) + getFilesForSubset(basePath_Paired, listClasses, include_train_patient)
  train_files_to_use_pet = getFilesForSubset(basePath_pet, listClasses, include_train_patient) + getFilesForSubset(basePath_Paired, listClasses, include_train_patient)
  print('TRAIN FILES MRI', len(train_files_to_use_mri))
  print('TRAIN FILES PET', len(train_files_to_use_pet))
  
  if enable_zoom:
    loader_s1_mri = readVolume_mri_zoom
    loader_s1_pet = readVolume_pet_zoom
  else:
    loader_s1_mri = readVolume_mri
    loader_s1_pet = readVolume_pet
    
  if enhance:
    loader_mri = lambda path: np_imadjust(loader_s1_mri(path), 0.005 , 0.995)
    loader_pet = lambda path: np_imadjust(loader_s1_pet(path), 0.005 , 0.995)
  else:
    loader_mri = loader_s1_mri
    loader_pet = loader_s1_pet
  
  mri_norm_range, min_p_m, max_p_m = DefineRangeNormalization(normalizzazione_range_mri, supLim_mri, infLim_mri, train_files_to_use_mri, loader_mri)
  sio.savemat(outputPath + 'Max_min_MRI.mat', {'min_p_m': min_p_m, 'max_p_m': max_p_m})
  
  pet_norm_range, min_p_p, max_p_p = DefineRangeNormalization(normalizzazione_range_pet, supLim_pet, infLim_pet, train_files_to_use_pet, loader_pet)
  sio.savemat(outputPath + 'Max_min_PET.mat', {'min_p_p': min_p_p, 'max_p_p': max_p_p})
  
  #-------------> CHECK Zscore Normalization STEP 4
  if zscore_mri:
    print('Computing mean and std')
    #np_computeMeanAndStd_all(train_files, readVolume, type_n, min_p, max_p, infLim, supLim)
    mean_v, std_v = np_computeMeanAndStd_all(train_files_to_use, loader_img, normalizzazione_range_mri, min_p_m, max_p_m, infLim_mri,supLim_mri)
    print('Mean MRI ', mean_v)
    print('STD MRI ', std_v)
    sio.savemat(outputPath + 'Mean_std_MRI.mat', {'mean_v': mean_v, 'std_v': std_v})
          
    train_transform_mri = transforms.Compose([
      ToTensor3D(),
      mri_norm_range,
      transforms.Normalize(mean_v, std_v),
      
      #tio.transforms.RandomBiasField((0,0.08), p=0.5),
      #tio.transforms.RandomBlur((0.0,0.8,0.0,0.8,0.0,0.8), p=0.5),
      #tio.transforms.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5),
      
      #random Translation
      tio.transforms.RandomAffine(scales=(1,1), degrees = (0,0,0,0,0,0), translation =(-10,10,-5,5,-10,10),isotropic  = True, default_pad_value  = 0, p=0.5),
      #random Rotation
      tio.transforms.RandomAffine(scales=(1,1), degrees = (-10,10,-5,5,-5,5), translation =(0,0,0,0,0,0),isotropic  = True, default_pad_value  = 0, p=0.5),
      #zoom
      tio.transforms.RandomAffine(scales=(0.9,1.1), degrees = (0,0,0,0,0,0), translation =(0,0,0,0,0,0),isotropic  = True, default_pad_value  = 0, p=0.5),
     
      Resize3D(size=(resize,resize,resize), enable_zoom = enable_zoom),
      ])
    
    val_transform_mri = transforms.Compose([
      ToTensor3D(),
      mri_norm_range,
      transforms.Normalize(mean_v, std_v),
      
      Resize3D(size=(resize,resize,resize), enable_zoom = enable_zoom),
      ])
      
  else:
    
    train_transform_mri = transforms.Compose([
      ToTensor3D(),
      mri_norm_range,
      
      #tio.transforms.RandomBiasField((0,0.08), p=0.5),
      #tio.transforms.RandomBlur((0.0,0.8,0.0,0.8,0.0,0.8), p=0.5),
      #tio.transforms.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5),
      
      #random Translation
      tio.transforms.RandomAffine(scales=(1,1), degrees = (0,0,0,0,0,0), translation =(-10,10,-5,5,-10,10),isotropic  = True, default_pad_value  = 0, p=0.5),
      #random Rotation
      tio.transforms.RandomAffine(scales=(1,1), degrees = (-10,10,-5,5,-5,5), translation =(0,0,0,0,0,0),isotropic  = True, default_pad_value  = 0, p=0.5),
      #zoom
      tio.transforms.RandomAffine(scales=(0.9,1.1), degrees = (0,0,0,0,0,0), translation =(0,0,0,0,0,0),isotropic  = True, default_pad_value  = 0, p=0.5),
     
      
      Resize3D(size=(resize,resize,resize), enable_zoom = enable_zoom),
      ])
    
    val_transform_mri = transforms.Compose([
      ToTensor3D(),
      mri_norm_range,
      Resize3D(size=(resize,resize,resize), enable_zoom = enable_zoom),
      ])
      
   
  if zscore_pet:
    print('Computing mean and std')
    #np_computeMeanAndStd_all(train_files, readVolume, type_n, min_p, max_p, infLim, supLim)
    mean_p, std_p = np_computeMeanAndStd_all(train_files_to_use, loader_img, normalizzazione_range_pet, min_p_m, max_p_m, infLim_pet,supLim_pet)
    print('Mean PET ', mean_p)
    print('STD PET ', std_p)
    sio.savemat(outputPath + 'Mean_std_PET.mat', {'mean_p': mean_p, 'std_p': std_p})
    
    train_transform_pet = transforms.Compose([
      ToTensor3D(),
      pet_norm_range,
      transforms.Normalize(mean_p, std_p),
      
      #random Translation
      tio.transforms.RandomAffine(scales=(1,1), degrees = (0,0,0,0,0,0), translation =(-10,10,-5,5,-10,10),isotropic  = True, default_pad_value  = 0, p=0.5),
      #random Rotation
      tio.transforms.RandomAffine(scales=(1,1), degrees = (-10,10,-5,5,-5,5), translation =(0,0,0,0,0,0),isotropic  = True, default_pad_value  = 0, p=0.5),
      #zoom
      tio.transforms.RandomAffine(scales=(0.9,1.1), degrees = (0,0,0,0,0,0), translation =(0,0,0,0,0,0),isotropic  = True, default_pad_value  = 0, p=0.5),
     
      Resize3D(size=(resize,resize,resize), enable_zoom = enable_zoom),
      ])
    
    val_transform_pet = transforms.Compose([
      ToTensor3D(),
      pet_norm_range,
      transforms.Normalize(mean_p, std_p),
      
      Resize3D(size=(resize,resize,resize), enable_zoom = enable_zoom),
      ])
      
  else:
    
    train_transform_pet = transforms.Compose([
      ToTensor3D(),
      pet_norm_range,
      
      #random Translation
      tio.transforms.RandomAffine(scales=(1,1), degrees = (0,0,0,0,0,0), translation =(-10,10,-5,5,-10,10),isotropic  = True, default_pad_value  = 0, p=0.5),
      #random Rotation
      tio.transforms.RandomAffine(scales=(1,1), degrees = (-10,10,-5,5,-5,5), translation =(0,0,0,0,0,0),isotropic  = True, default_pad_value  = 0, p=0.5),
      #zoom
      tio.transforms.RandomAffine(scales=(0.9,1.1), degrees = (0,0,0,0,0,0), translation =(0,0,0,0,0,0),isotropic  = True, default_pad_value  = 0, p=0.5),
     
      Resize3D(size=(resize,resize,resize), enable_zoom = enable_zoom),
      ])
    
    val_transform_pet = transforms.Compose([
      ToTensor3D(),
      pet_norm_range,
      Resize3D(size=(resize,resize,resize), enable_zoom = enable_zoom),
      ])
      
  
  num_classes = len(newClasses)
  print(train_transform_mri)
  print(train_transform_pet)
  print(val_transform_mri)
  print(val_transform_pet)
  
  
  dataset_paired = []
  
 
  for c in listClasses:
    is_valid_class = lambda path: c == path.split('/')[-2] 
    check_file = lambda path: include_train_patient(path) and is_valid_class(path)
    dataset_paired.append(My_DatasetFolderMultiModal(root= basePath_Paired,transform_mri= train_transform_mri, transform_pet= train_transform_pet, is_valid_file= check_file, list_classes = listClasses, readerMRI= loader_mri, readerPET= loader_pet))
  
  print('PAIRED ------')
  for i in range(0, len(listClasses)):
    print(listClasses[i] + ' PAIRED: ' + str(len(dataset_paired[i].samples)))
  
  '''  
  print('here')    
  dataset_mri[0].samples = sample(dataset_mri[0].samples, 3)
  dataset_mri[1].samples = sample(dataset_mri[1].samples, 3)
  dataset_mri[2].samples = sample(dataset_mri[2].samples, 3)
  
  print('here')    
  dataset_pet[0].samples = sample(dataset_pet[0].samples, 3)
  dataset_pet[1].samples = sample(dataset_pet[1].samples, 3)
  dataset_pet[2].samples = sample(dataset_pet[2].samples, 3)
  
  print('here')    
  dataset_paired[0].samples = sample(dataset_paired[0].samples, 3)
  dataset_paired[1].samples = sample(dataset_paired[1].samples, 3)
  dataset_paired[2].samples = sample(dataset_paired[2].samples, 3)
  completeTrainSet_paired = BalanceConcatDataset(dataset_paired)
  
  
  cn_dataset = dataset_paired[0]
  mci_dataset = dataset_paired[1]
  severe_dataset = dataset_paired[2]
  
  for p,l in cn_dataset.samples:
    if l != 0:
      print(p + ' Errore')
      
  
  for p,l in mci_dataset.samples:
    if l != 1:
      print(p + ' Errore')
  
  
  for p,l in severe_dataset.samples:
    if l != 2:
      print(p + ' Errore')
      
  '''
  completeTrainSet = BalanceConcatDataset(dataset_paired)
  print('After Balancing')
  for i in range(0, len(listClasses)):
    print(listClasses[i] + ' TOTAL : ' + str(len(dataset_paired[i].samples)))
    
  print('------------------------')
  val_paired_dataset = My_DatasetFolderMultiModal(root= basePath_Paired,transform_mri= val_transform_mri, transform_pet= val_transform_pet, is_valid_file= include_val_patient, list_classes = listClasses, readerMRI= loader_mri, readerPET= loader_pet)
  print('PAIRED - VALIDATION', len(val_paired_dataset.samples))
  
  print('------------------------')
  test_paired_dataset = My_DatasetFolderMultiModal(root= basePath_Paired,transform_mri= val_transform_mri, transform_pet= val_transform_pet, is_valid_file= include_test_patient, list_classes = listClasses, readerMRI= loader_mri, readerPET= loader_pet)
  print('PAIRED - TEST ', len(test_paired_dataset.samples))
  
  '''
  print('here')    
  val_mri_dataset.samples = sample(val_mri_dataset.samples, 3)
  val_pet_dataset.samples = sample(val_pet_dataset.samples, 3)
  val_paired_dataset.samples = sample(val_paired_dataset.samples, 3)
  
  print('here')    
  test_mri_dataset.samples = sample(test_mri_dataset.samples, 3)
  test_pet_dataset.samples = sample(test_pet_dataset.samples, 3)
  test_paired_dataset.samples = sample(test_paired_dataset.samples, 3)
  
  
  print('TRAIN FILES INFO . . . ')
  print(' CN CLASS ' + str(len(cn_dataset.samples)))
  print(' MILD CLASS ' + str(len(mci_dataset.samples)))
  print(' SEVERE CLASS ' + str(len(severe_dataset.samples)))
  '''
  
  model_conv = MyMedicalResNet(model_type, weightPath, num_classes, 2)
  model_conv = model_conv.to(device = device) 
  criterionCNN = nn.CrossEntropyLoss()

  loader_opts = {'batch_size': batchSize, 'num_workers': 0, 'pin_memory': False}
  
  optimizer_conv = optim.Adam(model_conv.parameters(), lr=learningRate, weight_decay= weightDecay)
  
  print('     Before Training: GPU Memory  %d bytes'%(torch.cuda.memory_allocated()))
  print('Parameters ',  count_parameters(model_conv))
  
  if not continue_learning:
    #inizializzazione senza check point
    best_acc = 0.0  
    best_acc_m = 0.0
    best_acc_w = 0.0    
    best_loss = 0.0 
    best_epoca = 0
    startEpoch = 1
  else:
    print('RELOAD')
    stato = sio.loadmat(outputPath + 'check_point.mat')
    best_acc = stato['best_acc'][0][0]
    best_acc_m = stato['best_acc_m'][0][0]
    best_acc_w = stato['best_acc_w'][0][0]
    best_loss = stato['best_loss'][0][0]
    best_epoca = stato['best_epoca'][0][0]
    startEpoch = stato['last_epoch'][0][0] + 1
    model_conv.load_state_dict(torch.load(outputPath + 'train_weights.pth'))
    
  model_conv = train_loop_validation(model_conv, 
                          completeTrainSet, val_paired_dataset, test_paired_dataset, 
                          startEpoch, num_epoch, 
                          loader_opts,
                          criterionCNN, optimizer_conv, 
                          best_acc, best_acc_m, best_acc_w, best_loss, best_epoca,
                          outputPath, device)
                          
  print('     After Training: GPU Memory  %d bytes'%(torch.cuda.memory_allocated()))
  model_conv.cpu()
  del model_conv
  print('     After Training: GPU Memory  %d bytes'%(torch.cuda.memory_allocated()))
    
  print('---------------------------------> BEST MODEL')
  lossModel_Train = []
  lossModel_val = []
  
  accModel_Train = []
  accModel_val = []

  Acc_0 = []
  Acc_1 = []

  file = open(outputPath + 'lossTrain.txt', 'r')
  Testo = file.readlines()
  for element in Testo:
    lossModel_Train.append(float(element))

  file = open(outputPath + 'lossVal.txt', 'r')
  Testo = file.readlines()
  for element in Testo:
    lossModel_val.append(float(element))

  plt.figure()
  plt.title("Model: Training Vs Validation Losses")
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.plot(list(range(1,len(lossModel_Train)+1)), lossModel_Train, color='r', label="Training Loss")  
  plt.plot(list(range(1, len(lossModel_val)+1)), lossModel_val, color='g', label="Validation Loss")
  plt.legend()
  plt.savefig(outputPath + 'LossTrainVal.png')


  file = open(outputPath + 'AccTrain.txt', 'r')
  Testo = file.readlines()
  for element in Testo:
    accModel_Train.append(float(element))

  file = open(outputPath + 'AccVal.txt', 'r')
  Testo = file.readlines()
  for element in Testo:
    accModel_val.append(float(element))

  plt.figure()
  plt.title("Training Vs Validation Accuracies")
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.plot(list(range(1, len(accModel_Train)+1)), accModel_Train, color='r', label="Training Accuracy")
  plt.plot(list(range(1, len(accModel_val)+1)), accModel_val, color='g', label="Validation Accuracy")
  plt.legend()
  plt.savefig(outputPath + 'AccTrainVal.png')
  
  
  file = open(outputPath + 'AccVal_0.txt', 'r')
  Testo = file.readlines()
  for element in Testo:
    Acc_0.append(float(element))

  file = open(outputPath + 'AccVal_1.txt', 'r')
  Testo = file.readlines()
  for element in Testo:
    Acc_1.append(float(element))
 

  plt.figure()
  plt.title("Validation Accuracies")
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.plot(list(range(1, len(Acc_0)+1)), Acc_0, color='r', label="Val Accuracy class 0")
  plt.plot(list(range(1, len(Acc_1)+1)), Acc_1, color='g', label="Val Accuracy class 1")
  plt.plot(list(range(1, len(accModel_val)+1)), accModel_val, color='b', label="Total Val Accuracy")
  plt.legend()
  plt.savefig(outputPath + 'AccVal.png')  
  
  
  model_conv = MyMedicalResNet(model_type, weightPath, num_classes, 2)
  model_conv.load_state_dict(torch.load(outputPath + 'best_model_weights.pth'))
  model_conv = model_conv.to(device = device) 

  model_conv.eval()
  #model_conv, test, transform
  tabella = prediction_on_Test(model_conv, test_paired_dataset, device)
  tabella.to_csv(outputPath + 'TabellaBestModel.csv', sep = ',', index=False)


  accuracy = np.sum(tabella.true_class.values == tabella.predicted.values)/tabella.shape[0]
  t0 = tabella[tabella.true_class == 0]
  t1 = tabella[tabella.true_class == 1]
  t2 = tabella[tabella.true_class == 2]

  accuracy_0 = np.sum(t0.true_class.values == t0.predicted.values)/t0.shape[0]
  accuracy_1 = np.sum(t1.true_class.values == t1.predicted.values)/t1.shape[0]
  accuracy_2 = np.sum(t2.true_class.values == t2.predicted.values)/t2.shape[0]
  
  print('Accuracy')
  print(accuracy)
  print('Accuracy_0')
  print(accuracy_0)
  print('Accuracy_1')
  print(accuracy_1)
  print('Accuracy_2')
  print(accuracy_2)

  model_conv.cpu()
  del model_conv
  print('     After Training: GPU Memory  %d bytes'%(torch.cuda.memory_allocated()))