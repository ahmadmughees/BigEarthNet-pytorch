import torch
from torch import sigmoid
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np
from torch.autograd import Variable


def accuracy(gt,pred):       
    gt = np.asarray(gt) #will round to the nearest even number
    pred = np.round(pred)      
    return{
    'f1_samples': f1_score(gt, pred, average = 'samples', zero_division=1),
    'f1_macro': f1_score(gt, pred, average = 'macro', zero_division=1),
    'f1_micro': f1_score(gt,pred,average = 'micro', zero_division=1)}

def validation(model, loader_, criterion):
    model.eval()
    test_iter=0
    losses = [] ; mean_losses = []
    with torch.no_grad():
        for batch_idx, (data, gt) in enumerate(loader_):
            data, gt = Variable(data.cuda()), Variable(gt.cuda())
            output = model(data)
            loss = criterion(output,gt)
            losses.append(loss.item())
            mean_losses.append(np.mean(losses[max(0,test_iter-100):test_iter]))

            pred= sigmoid(output).data.cpu().numpy()
            gt = gt.data.cpu().numpy()

            if test_iter==0:
                all_pred=pred
                all_gt=gt
            else:
                all_pred=np.vstack((all_pred,pred))
                all_gt  =np.vstack((all_gt,gt))

            test_iter+=1
        acc=accuracy(all_gt,all_pred)
        model.train()
    return acc, mean_losses

def testing(model, loader_,criterion):
    model.eval()
    test_iter=0
    with torch.no_grad():
        for batch_idx, (data, gt) in tqdm(enumerate(loader_)):
            data, gt = Variable(data.cuda()), Variable(gt.cuda())
            output = model(data)
            loss = criterion(output,gt)
            
            pred = sigmoid(output).data.cpu().numpy()
            gt = gt.data.cpu().numpy()
            if test_iter==0:
                all_pred=pred
                all_gt=gt
            else:
                all_pred=np.vstack((all_pred,pred))
                all_gt  =np.vstack((all_gt,gt))

            test_iter+=1
        # TODO DESIGN BETTER LOGGER 
        acc=accuracy(all_gt,all_pred)
        cm = multilabel_confusion_matrix(np.asarray(all_gt),np.round(all_pred))
        print('Test accuracy: {}'.format(acc))
        print(cm.reshape(19,4))

        model.train()
        return acc,cm