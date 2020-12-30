import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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

Batch_size = 256

# Model_name = 'base'
# epoch_weight_path = 'base/epoch_191_val_acc_0.807210_.pkl'

# Model_name ='weighted_loss_from_the_weights_of_base_low_lr'
# epoch_weight_path = 'weighted_loss_from_the_weights_of_base_low_lr/epoch_191_val_acc_0.696328_.pkl'

# Model_name = 'weighted_loss_from_the_weights_of_base'
# epoch_weight_path = 'weighted_loss_from_the_weights_of_base/epoch_191_val_acc_0.688985_.pkl' 

# Model_name = 'weighted_sampler_from_the_weights_of_base'
# epoch_weight_path = 'weighted_sampler_from_the_weights_of_base/epoch_178_val_acc_0.802138_.pkl' 

# Model_name = 'weighted_sampler_from_the_weights_of_base'
# epoch_weight_path = 'weighted_sampler_from_the_weights_of_base/epoch_179_val_acc_0.802008_.pkl' 

Model_name = 'weighted_sampler_from_the_weights_of_base'
epoch_weight_path = 'weighted_sampler_from_the_weights_of_base/epoch_183_val_acc_0.801986_.pkl' 

# Model_name = 'weighted_sampler_from_the_weights_of_base'
# epoch_weight_path = 'weighted_sampler_from_the_weights_of_base/epoch_185_val_acc_0.801947_.pkl' 

Model_name = 'weighted_sampler_and_loss_from_the_weights_of_base'
epoch_weight_path = 'weighted_sampler_and_loss_from_the_weights_of_base/epoch_179_val_acc_0.696578_.pkl'

Model_name = 'weighted_sampler_and_loss_from_the_weights_of_base'
epoch_weight_path = 'weighted_sampler_and_loss_from_the_weights_of_base/epoch_180_val_acc_0.695897_.pkl'

Model_name = 'weighted_sampler_and_loss_from_the_weights_of_base'
epoch_weight_path = 'weighted_sampler_and_loss_from_the_weights_of_base/epoch_192_val_acc_0.695803_.pkl'

Model_name = 'weighted_sampler_and_loss_from_the_weights_of_base'
epoch_weight_path = 'weighted_sampler_and_loss_from_the_weights_of_base/epoch_194_val_acc_0.695327_.pkl'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =============================================================================
# LOAD THE MODEL
# =============================================================================
   
def load_model():
    from torchvision import models
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,19)# len (LABELS))
    model.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding = 3, bias = False)
    print('Model Loaded')
    return model#.cuda()

# =============================================================================
# DATA_LOADER 
# =============================================================================

def data_loader(BATCH_SIZE = Batch_size):
    img_folder =  '../BEN/Images/'
    test_img_path = img_folder + 'Test/'
    
    lab_folder = '../BEN/GT/'
    test_lab_path = lab_folder + 'Test/'
    #tar = pd.read_csv('E:\Biglabelsjustclassnum.csv')
    test_data = BigEarthNet_Dataset(test_img_path,test_lab_path)

    test_loader=torch.utils.data.DataLoader(test_data,batch_size=BATCH_SIZE, num_workers=16,pin_memory = True)
    return {'test': test_loader}

#def main():
     #TODO will change every processing into one main function  


if __name__ == '__main__':
    #TODO write a function for seeding     
    torch.manual_seed(0)
    np.random.seed(0)
    print('pytorch version', torch.__version__)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    mkdir(os.path.join('./BEN_models', Model_name))
    loader = data_loader()
    model = load_model()
    #model.load_state_dict(torch.load('./BEN_models/base/epoch94_acc_0.747304'))
 
    load_checkpoint(model = model,
                     checkpoint_path = os.path.join('./BEN_models', epoch_weight_path))
    
    model.to(device)
    test_accuracy, conf = testing(model, loader['test'], epoch_weight_path[:-5])