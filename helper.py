import numpy as np
import matplotlib.pyplot as plt
from pytorch_msssim import ssim as calculate_ssim #, ms_ssim
from math import log10
import os
import torch 

def calculate_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def save_checkpoint(net=None, optimizer=None, epoch=None, train_losses=None, train_acc=None, val_loss = None, val_acc=None, check_loss=None, savepath=None, GPUdevices = 1):
    
    if GPUdevices > 1:
        net_weights = net.module.state_dict() 
    else:
        net_weights = net.state_dict()
    save_json = {
        'net_state_dict': net_weights,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_losses': train_losses,
        'train_acc': train_acc,
        'val_loss':val_loss,
        'val_acc': val_acc
    }
    if check_loss > val_loss[-1]:
        savepath = savepath + '/epoch_{}_val_acc_{:.6f}_best_params.pkl'.format(epoch,val_acc[-1])
        check_loss = val_loss[-1]
    else:
        savepath = savepath + '/epoch_{}_val_acc_{:.6f}_.pkl'.format(epoch,val_acc[-1])
    torch.save(save_json, savepath)
    print("checkpoint of {}th epoch saved at {}".format(epoch, savepath))

    return check_loss


def load_checkpoint(net, optimizer, checkpoint_path, best_params_flag):
    checkpoint = torch.load(checkpoint_path)

    net.load_state_dict(checkpoint['net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    losses = checkpoint['losses']

    return net, optimizer, start_epoch, losses

def logger_to_file():
    pass

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
   
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def show_time(now):
    s = str(now.year) + '/' + str(now.month) + '/' + str(now.day) + ' ' \
        + '%02d' % now.hour + ':' + '%02d' % now.minute + ':' + '%02d' % now.second
    return s
