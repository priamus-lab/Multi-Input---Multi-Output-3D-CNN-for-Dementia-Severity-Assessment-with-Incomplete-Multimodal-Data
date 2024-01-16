from TrainFunctions import*

#rendere l'esecuzione deterministica 
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print('backends.cudnn.version')
print(torch.backends.cudnn.version())
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print('backends.cudnn.version')
print(torch.version.cuda)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='1'


fold = 2
continue_learning = 'true'
tipofunzione = 'train'
normalizzazione_range_mri = 'range_intra'
normalizzazione_range_pet = 'range_intra'
zscore_mri = 'false'
zscore_pet = 'false'
supLim_mri = 1
infLim_mri = 0
supLim_pet = 1
infLim_pet = 0
enable_zoom = 'true'
enhance = 'false'
model_type = 'standard'

if continue_learning == 'true':
  continue_learning = True
else:
  continue_learning = False
  
if zscore_mri == 'true':
  zscore_mri = True
else:
  zscore_mri = False
  
if zscore_pet == 'true':
  zscore_pet = True
else:
  zscore_pet = False
  
if enable_zoom == 'true':
  enable_zoom = True
else:
  enable_zoom = False
  
if enhance == 'true':
  enhance = True
else:
  enhance = False
  
basePath = '/PATH/'  #your path
dfs = pd.read_excel(basePath + 'FoldForCV.xlsx', sheet_name='Sheet1')

basePath_mri = 'ONLY_MRI_final'
basePath_Paired = 'PAIRED_final'
basePath_pet = 'ONLY_PET_final'

listClasses = ['CN', 'MILD', 'SEVERE']
newClasses = ['CN', 'NO_CN']

ch = 16
learning_rate = 0.0001
w_deacy = 0.00001
batch_size = 32
numEpochs = 2
resize = 128


patient_to_include = dfs.Patient.to_list()  
print('Working on a set of Patients ' , len(patient_to_include))
test_set = dfs[dfs.FOLD == fold].Patient.to_list()
valFold = dfs[dfs.FOLD == fold].FOLD_VAL.values[0]
vali_set = dfs[dfs.FOLD == valFold].Patient.to_list()

if len(list(set(test_set) & set(vali_set)))>0:
  print('errore!!!')
  
print('Number of patients in validation ' + str(len(vali_set)))
print('Number of patients in test ' + str(len(test_set))) 

test_group = dfs[dfs.FOLD == fold]
val_group =  dfs[dfs.FOLD == valFold]

print('Validation info')
print_number_files(val_group)
print('Test info')
print_number_files(test_group)
  

outputPath = basePath + 'MultimodalEARLY_ch_' +str(ch) + '_CN_NOCN_' + normalizzazione_range_mri + '_P_' + normalizzazione_range_pet + '_'

if normalizzazione_range_mri != 'none':
  outputPath = outputPath + 'Sm_' + str(supLim_mri) + '_I_' + str(infLim_mri) + '_'

if zscore_mri:
  outputPath = outputPath + 'Zscorem_' 
  
if normalizzazione_range_pet != 'none':
  outputPath = outputPath + 'Sp_' + str(supLim_pet) + '_I_' + str(infLim_pet) + '_'

if zscore_pet:
  outputPath = outputPath + 'Zscorep_' 
  
if enable_zoom:
  outputPath = outputPath + 'Zoom_'
  
if enhance:
  outputPath = outputPath + 'Enhance_'

outputPath = outputPath + model_type + '/' + 'Fold' + str(fold) +'/'

try:
  os.makedirs(outputPath)
except OSError:
  pass
   
print(outputPath)

           
Main_train(vali_set, test_set, patient_to_include, listClasses, newClasses, basePath_mri, basePath_pet, basePath_Paired, ch, batch_size, outputPath,
           numEpochs, learning_rate, w_deacy, 
             continue_learning, model_type, resize,
             normalizzazione_range_mri, normalizzazione_range_pet, supLim_mri, infLim_mri, supLim_pet, infLim_pet, zscore_mri,
             zscore_pet, enable_zoom, enhance, device)

