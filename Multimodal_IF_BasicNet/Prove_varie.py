from TrainFunctionsMRI import*

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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


#fold = int(sys.argv[1])
#continue_learning = sys.argv[2]
#tipofunzione = sys.argv[3]

fold = 0
continue_learning = 'true'
tipofunzione = 'train'

if continue_learning == 'true':
  continue_learning = True
else:
  continue_learning = False
  
basePath = '/datadrive/Periodo_estero/'
dfs = pd.read_excel(basePath + 'DivisioneFOLD.xlsx', sheet_name='Sheet1')

basePath_mri = 'ONLY_MRI'
basePath_Paired = 'PAIRED'
listClasses = ['NORM', 'MCI', 'DEM']
newClasses = ['NORM', 'NO_NORM']

model_type = 'standard'
ch = 8
learning_rate = 0.0001
w_deacy = 0.00001
batch_size = 32
numEpochs = 20
normalizzazione = 'range'

test_set = dfs[dfs.FOLD == fold].Paziente.to_list()
valFold = dfs[dfs.FOLD == fold].FOLD_VAL.values[0]
vali_set = dfs[dfs.FOLD == valFold].Paziente.to_list()

if len(list(set(test_set) & set(vali_set)))>0:
  print('errore!!!')
  
  
 
train_transform = transforms.Compose([
  ToTensor3D(),
  #transforms.RandomRotation(degrees = 20),
  ])
    
f = Resize3D((256,256,256))
val_transform = transforms.Compose([
  ToTensor3D(),
])

if normalizzazione == 'zscore':
  loader_img = readVolume
else:
  loader_img = readVolume_range

pp = basePath + basePath_Paired + '/DEM/OAS30622_MR_d0802_PIB_d0802.mat'
x = loader_img(pp)
x1 = train_transform(x)
print(x1.shape)
print(torch.max(x1))
print(torch.min(x1))
xx = x1.permute(2,3,1,0)
plt.imshow(xx[:,:,128,0])
plt.show()

x1 = f(train_transform(x))
print(x1.shape)
print(torch.max(x1))
print(torch.min(x1))
xx = x1.permute(2,3,1,0)
plt.imshow(xx[:,:,128,0])
plt.show()


x1 = train_transform(x)
print(x1.shape)
print(torch.max(x1))
print(torch.min(x1))
xx = x1.permute(2,3,1,0)
plt.imshow(xx[:,:,128,0])
plt.show()


x = loadmat(pp)
y= x['norm'][0][0][0].astype(np.float32)
y = y[:,:,:, np.newaxis]
y = rangeNormalization(y, 1, 0)
x1 = train_transform(y)
print(x1.shape)
print(torch.max(x1))
print(torch.min(x1))
xx = x1.permute(2,3,1,0)
plt.imshow(xx[:,:,128,0])
plt.show()
'''
x1 = train_transform(x)
print(x1.shape)
print(torch.max(x1))
print(torch.min(x1))
xx = x1.permute(2,3,1,0)
plt.imshow(xx[:,:,128,0])
plt.show()

pp = basePath + basePath_Paired + '/DEM/OAS30850_MR_d0198_PIB_d0334.mat'
x = readVolume(pp)
x = train_transform(x)
print(x.shape)
print(torch.max(x))
print(torch.min(x))
xx = x.permute(2,3,1,0)

plt.imshow(xx[:,:,128,0])
plt.show()
'''