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

def print_stars():
    print("+++++++++++++++++++++++++++++++++++++++++")

def show_and_save_figure(gt, out, frame, folder_name = 'model'):
    #varn = var.numpy()
    gt = (np.transpose(gt.numpy(),[1,2,0])* 255.).astype(np.uint8)
    out =(np.transpose(out.numpy(),[1,2,0])* 255.).astype(np.uint8)
    video_folder,frame_label = frame.split('/')
    frame = video_folder + '_' + frame_label
    mkdir('./visual_results/' + folder_name)
    plt.subplot(121)
    plt.imshow(gt)
    plt.title('GT: {}'.format(frame))
    plt.imsave('./visual_results/'+folder_name+'/{}_GT.png'.format(frame),gt)
    plt.subplot(122)
    plt.imshow(out)
    plt.imsave('./visual_results/'+folder_name+'/{}_OUTPUT.png'.format(frame),out)
    plt.title('OUTPUT: {}'.format(frame))
    plt.show()

# def show_figure(gt, out, frame):
#     #varn = var.numpy()
#     plt.subplot(121)
#     plt.imshow((np.transpose(gt.numpy(),[1,2,0])* 255).astype(np.uint8))
#     plt.title('GT: {}'.format(frame))
#     plt.subplot(122)
#     plt.imshow((np.transpose(out.numpy(),[1,2,0])* 255).astype(np.uint8))
#     plt.title('OUTPUT: {}'.format(frame))
#     plt.show()

    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_psnr(_rmse, max_pixel = 1.0): 
    psnr = 20 * log10(max_pixel / _rmse) 
    return psnr 

def _PSNR(predictions, targets, max_pixel = 1.0): 
    _rmse = calculate_psnr(predictions, targets) 
    if(_rmse == 0):  # MSE is zero means no noise is present in the signal . 
                     # Therefore PSNR have no importance. 
        return 100
    psnr = 20 * log10(max_pixel / np.sqrt(_rmse)) 
    return psnr

def show_time(now):
    s = str(now.year) + '/' + str(now.month) + '/' + str(now.day) + ' ' \
        + '%02d' % now.hour + ':' + '%02d' % now.minute + ':' + '%02d' % now.second
    return s