

"""
clean code
val_sampler removed
RESNEXT50
with pos_weight of loss
With CSVread
With Random Sampler
With torch Rotate and flip 
with adam optimizer
with shuffle on
Created on Fri Oct 25 12:34:14 2019

@author: UBAID
"""
#import cv2
import numpy as np
from skimage import io
import random
from IPython.display import clear_output
import time
import os.path
from sklearn.metrics import f1_score
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable
import json
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix
from skimage.transform import resize

with open('label_indices.json', 'rb') as f:
    label_indices = json.load(f)
label_conversion = label_indices['label_conversion']
BigEarthNet_19_label_idx = {v: k for k, v in label_indices['BigEarthNet-19_labels'].items()}
original_labels_multi_hot = np.zeros(len(label_indices['original_labels'].keys()), dtype='float32')
BigEarthNet_19_labels_multi_hot = np.zeros(len(label_conversion),dtype='float32')


#from torch.optim.lr_scheduler import ReduceLROnPlatea
# Parameters
"""LABELS = [
    'Agro-forestry areas', 'Airports',
    'Annual crops associated with permanent crops', 'Bare rock',
    'Beaches, dunes, sands', 'Broad-leaved forest', 'Burnt areas',
    'Coastal lagoons', 'Complex cultivation patterns', 'Coniferous forest',
    'Construction sites', 'Continuous urban fabric',
    'Discontinuous urban fabric', 'Dump sites', 'Estuaries',
    'Fruit trees and berry plantations', 'Green urban areas',
    'Industrial or commercial units', 'Inland marshes', 'Intertidal flats',
    'Land principally occupied by agriculture, with significant areas of '
    'natural vegetation', 'Mineral extraction sites', 'Mixed forest',
    'Moors and heathland', 'Natural grassland', 'Non-irrigated arable land',
    'Olive groves', 'Pastures', 'Peatbogs', 'Permanently irrigated land',
    'Port areas', 'Rice fields', 'Road and rail networks and associated land',
    'Salines', 'Salt marshes', 'Sclerophyllous vegetation', 'Sea and ocean',
    'Sparsely vegetated areas', 'Sport and leisure facilities',
    'Transitional woodland/shrub', 'Vineyards', 'Water bodies', 'Water courses']"""
"""LABELS_WEIGHTS = 1 / torch.cuda.FloatTensor([30393, 756, 6940, 2882, 1534, 138035, 299, 1495, 103916,
                  138977, 1037, 10469, 64551, 718, 1050, 4652, 1614, 11428,
                  5244, 962, 122430, 3992, 151981, 5036, 11120, 177131, 12356,
                  98876, 15173, 13429, 429, 3757, 3249, 423, 1279, 11186,
                  71063, 1202, 4890, 135631, 9477, 44714, 9603])
class_weight =20/ torch.cuda.FloatTensor([22, 12, 9, 8, 33, 18, 29, 38, 20, 27, 3, 2, 11,
                                40, 32, 5, 14, 39, 1, 37, 42, 25, 4, 7, 13, 21,
                                6, 17, 43, 28, 41, 36, 31, 30, 16, 15, 10, 26,
                                34, 19, 23, 35, 24])"""
LABELS_WEIGHTS = torch.cuda.FloatTensor([1, 8, 2, 5, 8, 1, 10, 8, 1, 1, 8, 2, 1, 8, 8, 5, 8, 2, 5, 8, 1, 5, 1, 5, 2, 
                                         1, 2, 1, 1, 2, 10, 5, 5, 10, 8, 2, 1, 8, 5, 1, 2, 1, 2])

#mylabels = np.array(LABELS)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# instantiate the network     
#from torchvision import models, transforms
#model_ft = models.resnet18(pretrained=False)
#num_ftrs = model_ft.linear.in_features #for resnet
##num_ftrs = model_ft.classifier[-1].in_features #for vgg
##model_ft.classifier[-1] = nn.Linear(num_ftrs, len (LABELS)) #for vgg
#model_ft.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding = 3, bias = False) #for resnet
##model_ft.features[0] = nn.Conv2d(12, 64, kernel_size=3, stride=1, padding = 1, bias = False) #for vgg
#print('Model Loaded')
#net = model_ft.cuda()      
#del model_ft
from torchvision import models, transforms
model_ft = models.resnet18(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 19)
model_ft.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding = 3, bias = False)
print('Model Loaded')
net = model_ft.cuda()      
del model_ft
def accuracy(gt_S,pred_S):       
    gt_S  =np.asarray(gt_S) #will round to the nearest even number
    pred_S=np.round(pred_S)      
    f1s = f1_score(gt_S,pred_S,average = 'samples')
    f1m = f1_score(gt_S,pred_S,average = 'macro')
    print('f1MacroScore{}'.format(f1m))
    return f1s#np.mean(F1score)

def sigmoid(z):
    return 1/(1+np.exp(-z))

#import kornia
#kornia_transform = nn.Sequential(
#    kornia.color.AdjustBrightness(0.5),
#    kornia.color.AdjustGamma(gamma=2.),
#    kornia.color.AdjustContrast(0.7),
#)
def cus_aug(data):
    data = torch.rot90(data,random.randint(-3,3), dims=random.choice([[-1,-2],[-2,-1]]))
    #data = kornia_transform(data)
    if random.random()>0.5:
        data = torch.flip(data, dims = random.choice([[-2,],[-1,],[-2,-1]]))
    #else:
    #    data = kornia_transform(data)
    #pixmis = torch.randint_like(data, low=0, high=data.shape[3])
    #pixmis = torch.ones_like(data).random_(0,1)
    #pixmis = torch.empty_like(data).random_(0,1)
    #pixmis = torch.bernoulli(pixmis)
    
    #pixmis = torch.empty_like(data).random_(data.shape[-1])
    #pixmis = torch.where(pixmis>(data.shape[-1]/8),torch.ones_like(data),torch.zeros_like(data))
    
    #pixmis = torch.rand_like(data).bool()
    #pixmis = 1 if (pixmis>15) else 0
    #pixmis[pixmis<(data.shape[-1]/8)]=0
    #pixmis[pixmis>(data.shape[-1]/8)]=1
    #print(pixmis)
    return data#*pixmis 

def train(net, train_, val_, criterion, optimizer, epochs, scheduler=None, weights=None, save_epoch = 1):
    losses=[]; acc=[]; mean_losses=[]; val_acc=[]
    iter_ = t0 =0
    for e in range(1, epochs + 1):
        net.train()
        for batch_idx, (data, target) in enumerate(train_):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses = np.append(losses,loss.item())
            mean_losses = np.append(mean_losses, np.mean(losses[max(0,iter_-100):iter_]))
            
            if iter_ % 380 == 0:
                clear_output()
                print('Iteration Number',iter_,'{} seconds'.format(time.time() - t0))
                t0 = time.time()
                pred = output.data.cpu().numpy()#[0]
                pred=sigmoid(pred)
                gt = target.data.cpu().numpy()#[0]
                acc = np.append(acc,accuracy(gt,pred))
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}\tLearning Rate:{}'.format(
                    e, epochs, batch_idx, len(train_),
                    100. * batch_idx / len(train_), loss.item(), acc[-1],optimizer.param_groups[0]['lr']))
                plt.plot(mean_losses) and plt.show()
                val_acc = np.append(val_acc,validation(net, val_))
                print('validation accuracy : {}'.format(val_acc[-1]))
                plt.plot( range(len(acc)) ,acc,'b',label = 'training')
                plt.plot( range(len(val_acc)), val_acc,'r--',label = 'validation')
                plt.legend() and plt.show()
                #print(mylabels[np.where(gt[1,:])[0]])
            iter_ += 1
            
            del(data, target, loss)
        if scheduler is not None:
           scheduler.step()
        if e % save_epoch == 0:
            
            torch.save(net.state_dict(), '.\Big_Resnet18_19class{}'.format(e))
    return net
def sampler_(labels):
    _, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    sample_weights = weights[labels]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return sampler
def loader(BATCH_SIZE = 512,train_split=0.60):
    #IMAGE_FOLDER = 'E:\BigEarthNet\Images/'
    train_IMAGE_FOLDER = r'E:\NewBigEarthNet\train/'
    val_IMAGE_FOLDER = r'E:\NewBigEarthNet\val/'
    test_IMAGE_FOLDER = r'E:\NewBigEarthNet\test/'
    
    train_LABEL_FOLDER = r'E:\UpdatedLabelsasdefinedsplits\train/'
    val_LABEL_FOLDER = r'E:\UpdatedLabelsasdefinedsplits\val/'
    test_LABEL_FOLDER = r'E:\UpdatedLabelsasdefinedsplits\test/'
    #tar = pd.read_csv('E:\Biglabelsjustclassnum.csv')
    train_data = BigEarthNet_dataset(train_IMAGE_FOLDER,train_LABEL_FOLDER)
    val_data = BigEarthNet_dataset(val_IMAGE_FOLDER,val_LABEL_FOLDER)
    test_data = BigEarthNet_dataset(test_IMAGE_FOLDER,test_LABEL_FOLDER)
    #tar_f=[item for sublist in tar_f for item in sublist]
    dataset_size = len(train_data)
    indices = list(range(dataset_size))
    #np.random.shuffle(indices) # shuffle the dataset before splitting into train and val
    #split = int(np.floor(train_split * dataset_size))
    #train_indices, val_indices, test_indices = indices[:split], indices[split:split+15000],indices[split+15000:]
    #train_labels = [tar.Label[x] for x in train_indices]
    #val_labels = [tar.Label[x] for x in val_indices]
    #train_sampler, val_sampler = sampler_(train_labels), sampler_(val_labels)
    #train_sampler = sampler_(train_labels)
    #b_train_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size = BATCH_SIZE, drop_last=True)
    train_loader=torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle = True,num_workers=3, pin_memory = True)
    #b_val_sampler = torch.utils.data.BatchSampler(val_sampler, batch_size = BATCH_SIZE, drop_last=True)    
    #b_val_sampler = torch.utils.data.SequentialSampler(val_labels)
    
    val_loader=torch.utils.data.DataLoader(val_data,batch_size=BATCH_SIZE, shuffle = True, num_workers=1,pin_memory = True)
    test_loader=torch.utils.data.DataLoader(test_data,batch_size=BATCH_SIZE, num_workers=0,pin_memory = True)
    #train_loader=torch.utils.data.DataLoader(torch.utils.data.Subset(data,train_indices),batch_size=BATCH_SIZE, batch_sampler=b_train_sampler, num_workers=4, pin_memory = True, sampler = train_sampler)
    #val_loader=torch.utils.data.DataLoader(torch.utils.data.Subset(data,val_indices),batch_size=BATCH_SIZE,batch_sampler=b_val_sampler, num_workers=3,pin_memory = True, sampler = val_sampler)
    return train_loader, val_loader,test_loader
def validation(model, test_,):
    model.eval()
    #tot_acc=[]
    test_iter=0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            output = model(data)
            #_, preds = torch.max(outputs, 1)
            pred = output.data.cpu().numpy()#[0]
            pred=sigmoid(pred)
            gt = target.data.cpu().numpy()
            if test_iter==0:
                all_pred=pred
                all_gt=gt
            else:
                all_pred=np.vstack((all_pred,pred))
                all_gt  =np.vstack((all_gt,gt))

            test_iter=test_iter+1
        acc=accuracy(all_gt,all_pred)
        #cm = multilabel_confusion_matrix(np.asarray(all_gt),np.round(all_pred))
        #print('Test accuracy: {}'.format(acc))
        #print(cm.reshape(43,4))
        #print(mylabels[np.where(gt[1,:])[0]]) #printing labels
        #print(mylabels[np.where(np.round(pred[1,:]))[0]]) #printing labels
        model.train()
        return acc#,cm

def test(model, test_,):
    model.eval()
    #tot_acc=[]
    test_iter=0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            output = model(data)
            #_, preds = torch.max(outputs, 1)
            pred = output.data.cpu().numpy()#[0]
            pred=sigmoid(pred)
            gt = target.data.cpu().numpy()
            if test_iter==0:
                all_pred=pred
                all_gt=gt
            else:
                all_pred=np.vstack((all_pred,pred))
                all_gt  =np.vstack((all_gt,gt))

            test_iter=test_iter+1
        acc=accuracy(all_gt,all_pred)
        cm = multilabel_confusion_matrix(np.asarray(all_gt),np.round(all_pred))
        print('Test accuracy: {}'.format(acc))
        print(cm.reshape(43,4))
        #print(mylabels[np.where(gt[1,:])[0]]) #printing labels
        #print(mylabels[np.where(np.round(pred[1,:]))[0]]) #printing labels
        model.train()
        return acc,cm
# Get a batch of training data
#inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
#out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])
def get_random_pos(img, window_shape = [100,100] ):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    #x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    #y2 = y1 + h
    return x1, x1+w, y1, y1+h#x1, x2, y1, y2

def random_crop_area(img):
    x1,x2,y1,y2 = get_random_pos(img)
    Sen_Im = img[:, x1:x2,y1:y2]
    return resize(Sen_Im,img.shape,anti_aliasing=True)
a_mean = np.array([340.76769064, 429.9430203, 614.21682446, 590.23569706, 950.68368468, 1792.46290469,
                             2075.46795189,2218.94553375,2266.46036911,2246.0605464,1594.42694882,1009.32729131],dtype='float32').reshape(12,1,1)
a_std = np.array([554.81258967,572.41639287,582.87945694,675.88746967,729.89827633,1096.01480586,
                     1273.45393088,1365.45589904,1356.13789355,1302.3292881,1079.19066363,818.86747235],dtype='float32').reshape(12,1,1)
class BigEarthNet_dataset(torch.utils.data.Dataset):
    def __init__(self,IMAGE_FOLDER,LABEL_FOLDER, cache=False, transforms=transforms):
        super(BigEarthNet_dataset, self).__init__()
        self.Image_folder_dir = os.listdir(IMAGE_FOLDER)
        self.LABEL_folder_dir = os.listdir(LABEL_FOLDER)
        self.indices=list(range(len(self.Image_folder_dir)))
        self.IMAGE_FOLDER = IMAGE_FOLDER
        self.LABEL_FOLDER=LABEL_FOLDER
    def __getitem__(self, Im_Id):
        Im_Files=self.IMAGE_FOLDER +'/'+self.Image_folder_dir[self.indices[Im_Id]]
        Sen_Im = np.asarray((io.imread(Im_Files)),dtype='float32')#/32000
        Sen_Im = Sen_Im.transpose(2,0,1)#/Sen_Im.max()
        Sen_Im = Sen_Im-a_mean
        Sen_Im = Sen_Im/a_std
        #if random.random() > 0.3:
        #    Sen_Im = random_crop_area(Sen_Im)
        
        lab=np.zeros((43),dtype='float32')
        Json_File=self.LABEL_FOLDER+'/' +self.LABEL_folder_dir[self.indices[Im_Id]]
        json_data = json.load(open(Json_File, 'r'))
        original_labels=json_data['labels']
 
        for label in original_labels:
            original_labels_multi_hot[label_indices['original_labels'][label]] = 1
        
        for i in range(len(label_conversion)):
            BigEarthNet_19_labels_multi_hot[i] = (
                    np.sum(original_labels_multi_hot[label_conversion[i]]) > 0).astype('float32')

        #BigEarthNet_19_labels = []
        #for i in np.where(BigEarthNet_19_labels_multi_hot == 1)[0]:
        #    BigEarthNet_19_labels.append(BigEarthNet_19_label_idx[i])
 
        # Return the torch.Tensor values
        return (torch.from_numpy(Sen_Im), #torplot ch.from_numpy(Sen_Im).todevice(device)
                torch.from_numpy(BigEarthNet_19_labels_multi_hot))
    def __len__(self):
        return len(self.Image_folder_dir)
if __name__ == '__main__':#https://discuss.pytorch.org/t/brokenpipeerror-errno-32-broken-pipe-when-i-run-cifar10-tutorial-py/6224/4
    #net = ResNet34() #Mughees: Just use Resnet18 or Resnet34     
    #torch.manual_seed(0)
    np.random.seed(0)
    #torch.cuda.manual_seed(0)
    #np.random.seed(0)
    #random.seed(0)
    train_loader,val_loader, test_loader = loader()
    #data = BigEarthNet_dataset(IMAGE_FOLDER,LABEL_FOLDER)
    #targetU = data.target
    #class_sample_count = np.unique(targetU, return_counts=True)[1]
    #weight = 1. / class_sample_count
    #samples_weight = weight[targetU]
    #samples_weight = torch.from_numpy(tar_lab)
    #del tar_lab
    #sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
    #data_split=0.70
    #L_data = len(data)
    #lengths = [int(L_data*data_split), 25000 , L_data - int(L_data*(data_split))-25000]
    #train_set, val_set, test_set = torch.utils.data.random_split(data,lengths)
    
    
    criteria=nn.BCEWithLogitsLoss()#pos_weight = LABELS_WEIGHTS)
    #base_lr = 0.001
    
    #optimizer = optim.Adam(net.parameters(), lr=base_lr)
    # We define the scheduler
    net=net.cuda() 
    #net.load_state_dict(torch.load('./Big_Resnet18_withloss57'))
    #optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.90, weight_decay=0.0005)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,60,65], gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=0.1,cycle_momentum=True)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=False)
    net = train(net, train_loader, val_loader, criteria, optimizer,70,scheduler)
    #test_loader=torch.utils.data.DataLoader(test_set,batch_size=BATCH_SIZE)
    test_accuracy,conf = test(net, test_loader)
    #print('Test Accuracy: {}'.format(test_accuracy))       
#acc = test(net, test_ids, all=False, stride=min(WINDOW_SIZE))
#train(net, optimizer,2, scheduler)    
#test(net, test_ids, all=False, stride=min(WINDOW_SIZE))

#net.load_state_dict(torch.load('./segnet_final'))
#_, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py