import torch
from torch import sigmoid
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np
from torch.autograd import Variable
import os
from helper import show_time
import datetime
from metrics import MetricTracker, Precision_score, Recall_score, F1_score, F2_score, Hamming_loss, Subset_accuracy, \
    Accuracy_score, One_error, Coverage_error, Ranking_loss, LabelAvgPrec_score, calssification_report

BigEarthNet_19_labels = ["Urban fabric","Industrial or commercial units","Arable land","Permanent crops","Pastures",\
        "Complex cultivation patterns","Land principally occupied by agriculture, with significant areas of natural vegetation",\
        "Agro-forestry areas","Broad-leaved forest","Coniferous forest",\
        "Mixed forest","Natural grassland and sparsely vegetated areas",\
        "Moors, heathland and sclerophyllous vegetation","Transitional woodland, shrub", "Beaches, dunes, sands",\
        "Inland wetlands","Coastal wetlands","Inland waters","Marine waters"]

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
        for batch_idx, (data, gt) in tqdm(enumerate(loader_)):
            data, gt = Variable(data.cuda()), Variable(gt.cuda())
            output = model(data)
            loss = criterion(output,gt)
            losses.append(loss.item())
            mean_losses.append(np.mean(losses[max(0,test_iter-100):test_iter])) #TODO remove the dependency from train_iter

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


prec_score_ = Precision_score()
recal_score_ = Recall_score()
f1_score_ = F1_score()
f2_score_ = F2_score()
hamming_loss_ = Hamming_loss()
subset_acc_ = Subset_accuracy()
acc_score_ = Accuracy_score()
one_err_ = One_error()
coverage_err_ = Coverage_error()
rank_loss_ = Ranking_loss()
labelAvgPrec_score_ = LabelAvgPrec_score()
calssification_report_ = calssification_report(BigEarthNet_19_labels)

def testing(model, loader_,Model_name):
    model.eval()
    test_iter=0
    with torch.no_grad():
        for batch_idx, (data, gt) in tqdm(enumerate(loader_)):
            data, gt = Variable(data.cuda()), Variable(gt.cuda())
            output = model(data)
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
        predicted_probs = np.asarray(all_pred)
        y_predicted = (predicted_probs >= 0.5).astype(np.float32)
        y_true = np.asarray(all_gt)
        
        macro_f1, micro_f1, sample_f1 = f1_score_(y_predicted, y_true)
        macro_f2, micro_f2, sample_f2 = f2_score_(y_predicted, y_true)
        macro_prec, micro_prec, sample_prec = prec_score_(y_predicted, y_true)
        macro_rec, micro_rec, sample_rec = recal_score_(y_predicted, y_true)
        hamming_loss = hamming_loss_(y_predicted, y_true)
        subset_acc = subset_acc_(y_predicted, y_true)
        macro_acc, micro_acc, sample_acc = acc_score_(y_predicted, y_true)
    
        one_error = one_err_(predicted_probs, y_true)
        coverage_error = coverage_err_(predicted_probs, y_true)
        rank_loss = rank_loss_(predicted_probs, y_true)
        labelAvgPrec = labelAvgPrec_score_(predicted_probs, y_true)
    
        cls_report = calssification_report_(y_predicted, y_true)
        print('Coastal wetlands:', cls_report['Coastal wetlands'])
        print('Beaches, dunes, sands',cls_report['Beaches, dunes, sands'])
        print('industrial and commercial',cls_report['Industrial or commercial units'])
        
        fp = open('testing_information.txt', 'a+')
        print('{}\t{}\tCoastal wetlands: Precision{:.6f}\Recall{:.6f}\F1_score{:.6f}\tBeaches, dunes, sands: Precision{:.6f}\Recall{:.6f}\F1_score{:.6f}\tindustrial and commercial: Precision{:.6f}\Recall{:.6f}\F1_score{:.6f}\t              Overall: F1_samples: {:.6f}\tF1_micro:{:.6f}\tF1_macro:{:.6f}'.format(
                show_time(datetime.datetime.now()), Model_name, 
                cls_report['Coastal wetlands']['precision'], cls_report['Coastal wetlands']['recall'], cls_report['Coastal wetlands']['f1-score'],
                cls_report['Beaches, dunes, sands']['precision'], cls_report['Beaches, dunes, sands']['recall'], cls_report['Beaches, dunes, sands']['f1-score'],
                cls_report['Industrial or commercial units']['precision'], cls_report['Industrial or commercial units']['recall'], cls_report['Industrial or commercial units']['f1-score'],
                acc['f1_samples'], acc['f1_micro'], acc['f1_macro']),file=fp)    
        fp.close()
        
        just_values = open('testing_information_just_values.csv', 'a+')
        print('{},\t{},\t{:.6f},{:.6f},{:.6f},\t{:.6f},{:.6f},{:.6f},\t{:.6f},{:.6f},{:.6f},\t{:.6f},{:.6f},{:.6f}'.format(
                show_time(datetime.datetime.now()), Model_name, 
                cls_report['Coastal wetlands']['precision'], cls_report['Coastal wetlands']['recall'], cls_report['Coastal wetlands']['f1-score'],
                cls_report['Beaches, dunes, sands']['precision'], cls_report['Beaches, dunes, sands']['recall'], cls_report['Beaches, dunes, sands']['f1-score'],
                cls_report['Industrial or commercial units']['precision'], cls_report['Industrial or commercial units']['recall'], cls_report['Industrial or commercial units']['f1-score'],
                acc['f1_samples'], acc['f1_micro'], acc['f1_macro']),file=just_values)    
        just_values.close()
        
        info = {
            "macroPrec" : macro_prec,
            "microPrec" : micro_prec,
            "samplePrec" : sample_prec,
            "macroRec" : macro_rec,
            "microRec" : micro_rec,
            "sampleRec" : sample_rec,
            "macroF1" : macro_f1,
            "microF1" : micro_f1,
            "sampleF1" : sample_f1,
            "macroF2" : macro_f2,
            "microF2" : micro_f2,
            "sampleF2" : sample_f2,
            "HammingLoss" : hamming_loss,
            "subsetAcc" : subset_acc,
            "macroAcc" : macro_acc,
            "microAcc" : micro_acc,
            "sampleAcc" : sample_acc,
            "oneError" : one_error,
            "coverageError" : coverage_error,
            "rankLoss" : rank_loss,
            "labelAvgPrec" : labelAvgPrec,
            "clsReport": cls_report
        }
        save_file_name = open(os.path.join('./BEN_models',Model_name+'_testing_information.txt'), 'a+')

        print("saving metrics...")
        print(info,file=save_file_name )
        save_file_name.close()
        #np.save('weightedLossandsampler_metrics.npy', info)
        return acc,cm