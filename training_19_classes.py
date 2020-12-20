import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import numpy as np
import random
from IPython.display import clear_output
import os.path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable
from torch import sigmoid
from tqdm import tqdm
from dataset import BigEarthNet_Dataset
from evaluation import testing, validation, accuracy
from helper import mkdir, save_checkpoint, show_time, load_checkpoint
import datetime
# =============================================================================
# HYPER_PARAMETERS
# =============================================================================

Start_epoch = 1   #incase of loading previouys weights
Lr = 0.001               # learn_rate
Drop_LR_at_epochs = [100,150,175,190]  # multistep scheduler
Epochs = 200            # no of epochs  
Milestones = [i - Epoch_start for i in Drop_LR_at_epochs]
Batch_size = 256 * 2 
Model_name = 'weighted_loss'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# instantiate the network     
def load_model():
    from torchvision import models
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,19)# len (LABELS))
    model.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding = 3, bias = False)
    print('Model Loaded')
    
    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)#,device_ids=[1, 2, 3])
    model.to(device)
    return model#.cuda()

def weight_for_weightedBCEloss():
    samples = torch.FloatTensor([74891,11865,194148,98997,29350,104203,130637,30649,141300,164775,176567,16267,148950,1536,12022,22100,1566,67277,74877])
    a = torch.FloatTensor(1) 
    step1 = torch.log2(torch.max(samples)/samples)
    return torch.max(a,step1)
# tensor([1.3743, 4.0324, 1.0000, 1.0000, 2.7257, 1.0000, 1.0000, 2.6632, 1.0000,
#         1.0000, 1.0000, 3.5771, 1.0000, 6.9818, 4.0134, 3.1350, 6.9539, 1.5290,
#         1.3746])

def cus_aug(data):
    data = torch.rot90(data,random.randint(-3,3), dims=random.choice([[-1,-2],[-2,-1]]))
    if random.random()>0.5:
        data = torch.flip(data, dims = random.choice([[-2,],[-1,],[-2,-1]]))
    return data

def train_one_epoch(model=None, loader=None, criterion=None, optimizer=None):
    losses = []; acc = []; mean_losses = []
    iter_ = 0
    model.train()
    for batch_idx, (data, gt) in tqdm(enumerate(loader['train'])):
        data, gt = cus_aug(Variable(data.cuda())), Variable(gt.cuda())
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, gt)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        mean_losses.append(np.mean(losses[max(0,iter_-100):iter_]))
        iter_ += 1
    clear_output()
    pred = sigmoid(output)                
    pred = pred.data.cpu().numpy()#[0]
    gt = gt.data.cpu().numpy()#[0]
    acc_dict = accuracy(gt,pred)
    acc.append(acc_dict['f1_micro'])
        

    return {'model': model,
            'train_accuracy': acc,
            'train_loss': losses,
            'mean_train_losses': mean_losses}

def training(epochs=None, model=None, loader=None, criterion=None, optimizer=None, scheduler= None, check_loss = 10.):
    train_accuracy = 0; train_losses = 0;  val_accuracy = 0
    mean_losses = 0; val_losses = 0
    for e in range(Start_epoch, epochs):
        #print("training of {}th epoch started.".format(e))
        after_epoch = train_one_epoch(model=model, loader=loader, criterion=criterion, optimizer=optimizer)
        
        if scheduler is not None:
            scheduler.step()
        val_acc_dict, val_loss = validation(after_epoch['model'], loader['val'], criterion)
        val_acc = val_acc_dict['f1_micro']
        train_accuracy = np.hstack((train_accuracy, np.array(after_epoch['train_accuracy'])))
        train_losses = np.hstack((train_losses, np.array(after_epoch['train_loss'])))
        mean_losses = np.hstack((mean_losses, np.array(after_epoch['mean_train_losses'])))
        val_accuracy = np.hstack((val_accuracy, val_acc))
        val_losses = np.hstack((val_losses, val_loss))
        #print(mean_losses)

        print('{}\tTrain (epoch {}/{}) \tTrain_Loss: {:.6f}\ttrain_acc: {:.6f}\tval_loss: {:.6f}\tval_acc: {:.6f}\tLR:{}'.format(
                show_time(datetime.datetime.now()), e, epochs, train_losses[-1], train_accuracy[-1], val_losses[-1],\
                val_accuracy[-1], scheduler.get_last_lr()))    

        check_loss = save_checkpoint(net = after_epoch['model'],
                        optimizer = optimizer,
                        epoch = e,
                        train_losses = mean_losses,
                        train_acc = train_accuracy, 
                        val_loss = val_losses, 
                        val_acc = val_accuracy, 
                        check_loss = check_loss, 
                        savepath = os.path.join('./BEN_models', Model_name),
                        GPUdevices = torch.cuda.device_count())

        plt.plot( range( len(mean_losses)), mean_losses, 'b',label = 'training_loss'), plt.show()
        print('validation accuracy : {:.6f}'.format(val_accuracy[-1]))
        plt.plot( range( len(train_accuracy)) ,train_accuracy,'b',label = 'training_acc')
        plt.plot( range( len(val_accuracy)), val_accuracy,'r--',label = 'validation_acc')
        plt.legend(), plt.show()
        fp = open(os.path.join('./BEN_models',Model_name, Model_name+'_training_information.txt'), 'a+')
        print('{}\tTrain (epoch {}/{}) \tTrain_Loss: {:.6f}\ttrain_acc: {:.6f}\tval_loss: {:.6f}\tval_acc: {:.6f}\tLR:{}'.format(
                show_time(datetime.datetime.now()), e, epochs, train_losses[-1], train_accuracy[-1], val_losses[-1],\
                val_accuracy[-1], scheduler.get_last_lr()),file=fp)    
        fp.close()
#        if torch.cuda.device_count()>1:
#            torch.save(after_epoch['model'].module.state_dict(), 
#                   os.path.join('./BEN_models', Model_name, 'epoch{}_acc_{:.6f}'.format(e, val_acc[-1])))
#        else:
#            torch.save(after_epoch['model'].state_dict(), 
#                   os.path.join('./BEN_models', Model_name, 'epoch{}_acc_{:.6f}'.format(e, val_acc[-1])))
    return after_epoch['model']


def data_loader(BATCH_SIZE = Batch_size):
    img_folder =  '../BEN/Images/'
    train_img_path = img_folder + 'Train/'
    val_img_path = img_folder + 'Val/'
    test_img_path = img_folder + 'Test/'
    
    lab_folder = '../BEN/GT/'
    train_lab_path = lab_folder + 'Train/'
    val_lab_path = lab_folder + 'Val/'
    test_lab_path = lab_folder + 'Test/'
    #tar = pd.read_csv('E:\Biglabelsjustclassnum.csv')
    train_data = BigEarthNet_Dataset(train_img_path,train_lab_path)
    val_data = BigEarthNet_Dataset(val_img_path,val_lab_path)
    test_data = BigEarthNet_Dataset(test_img_path,test_lab_path)

    train_loader=torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True, num_workers=32, pin_memory = True)    
    val_loader=torch.utils.data.DataLoader(val_data,batch_size=BATCH_SIZE, shuffle = True, num_workers=4,pin_memory = True)
    test_loader=torch.utils.data.DataLoader(test_data,batch_size=BATCH_SIZE, num_workers=0,pin_memory = True)
    return {'train': train_loader,
            'val': val_loader,
            'test': test_loader}



# Get a batch of training data
#inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
#out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])
if __name__ == '__main__':
    #TODO write a function for seeding     
    torch.manual_seed(0)
    np.random.seed(0)
    print(torch.__version__)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    mkdir(os.path.join('./BEN_models', Model_name))
    loader = data_loader()
    #criteria = torch.nn.MultiLabelSoftMarginLoss()
    #criteria = torch.nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss(pos_weight = weight_for_weightedBCEloss().to(device))
    model = load_model()
    #model.load_state_dict(torch.load('/media/hmahmad/Data/BigEarthNetCodes/BEN_models/base/epoch94_acc_0.747304'))
    optimizer = optim.Adam(model.parameters(), lr=Lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=Milestones, gamma=0.1)
    
    epoch_weight_path = None
    model, optimizer, Start_epoch = load_checkpoint(model = model,
                    optimizer=optimizer, 
                    checkpoint_path = os.path.join('./BEN_models', Model_name, epoch_weight_path))
    
    
    model = training(epochs=Epochs, model=model, loader=loader, criterion=criterion, optimizer=optimizer, scheduler=scheduler)
    test_accuracy, conf = testing(model, loader['test'], criterion)