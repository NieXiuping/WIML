from math import exp, isnan
from operator import ne
from numpy.lib.shape_base import expand_dims
import torch
from torch._C import DictType
from torchvision.transforms import transforms as T
import argparse 
import network.network_FPSNet as network
from scipy import ndimage
from torch import optim
from dataloader.dataset import TestingDataset
from numpy import random
from torch.utils.data import DataLoader, dataloader
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn 
import torch.nn.functional as F
import torch.nn as nn
import cv2
import numpy as np
import colorama
from colorama import Fore, Back, Style
from sklearn.manifold import TSNE
import os
from utils import largestConnectComponent
import time
from tensorboardX import SummaryWriter
from warnings import simplefilter
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=DeprecationWarning)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print(device)

x_transform = T.Compose([
    T.ToTensor(), 
    T.Resize(size=(160,160)),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

y_transform = T.Compose([
    T.ToTensor(),
    T.Resize(size=(160,160))
])

def tester(model, criterion_s, criterion_c, dataloader, device, imagesave_dir):

    print("Test Model".center(60, '-'))
    test_seg_loss_sum, test_cls_loss_sum = [], []
    scores, preds, labels = [], [], []
    TP_sum, FP_sum, FN_sum = [], [], []
    correct = 0
    step = 0
    confusion_matrix_class = torch.zeros(2, 2)

    model.eval()

    with torch.no_grad():

        for x_c, y_c, y_s, img_name in dataloader:
            
            ## Input image
            x_c = x_c.to(device)
            y_c = y_c.to(device)
            y_s = y_s.to(device)
            out_s, out_c = model(x_c)
            
            ## Compute loss
            loss_s_f = criterion_s(out_s, y_s) #segmentation loss
            test_seg_loss_sum.append(loss_s_f.item())
            loss_s = np.mean(np.asarray(test_seg_loss_sum))
            
            pred = torch.argmax(out_c)

            out_c = out_c.squeeze(2).squeeze(2)        
            loss_c_f = criterion_c(out_c, y_c) #classification loss
            test_cls_loss_sum.append(loss_c_f.item())
            loss_c = np.mean(np.asarray(test_cls_loss_sum))

            ## Compute class metrics
            for t, p in zip(y_c.view(-1), pred.view(-1)):
                confusion_matrix_class[t.long(), p.long()] += 1

            scores.append(F.softmax(out_c, dim=1)[:,1].cpu())
            preds.append(pred.view(-1).cpu())
            labels.append(y_c.view(-1).cpu())
            correct += pred.eq(y_c.view_as(pred)).sum().item() 

            ## Compute seg metrics
            out1 = torch.sigmoid(out_s)
            out1 = out1.cpu()
            out1[out1 >= 0.5] = 1
            out1[out1 < 0.5] = 0

            ## save test results
            out2 = out1.clone().numpy().squeeze() 
            out2 = cv2.resize(out2,(448, 448))
            out2 = largestConnectComponent(out2)
            strt = ndimage.generate_binary_structure(2, 1)
            out2 = np.asarray(
                ndimage.morphology.binary_opening(out2, strt), np.uint8)
            out2 = np.asarray(
                ndimage.morphology.binary_closing(out2, strt), np.uint8)
            out2 = ndimage.binary_fill_holes(out2)
            path_save = os.path.join(imagesave_dir,str(img_name[0]))                                                                                                                                                
            cv2.imwrite(path_save, out2*255)
        
            label = y_s.clone()
            label = label.cpu()
            label[label >= 0.5] = 1
            label[label < 0.5] = 0
            TP = np.sum(np.logical_and(out1.numpy().astype(np.int), label.numpy().astype(np.int)))
            FP = np.sum(np.logical_and(out1.numpy().astype(np.int), 
                            np.logical_not(label.numpy().astype(np.int))))
            FN = np.sum(np.logical_and(np.logical_not(out1.numpy().astype(np.int)), 
                            label.numpy().astype(np.int)))
            TP_sum.append(TP)
            FP_sum.append(FP)
            FN_sum.append(FN)

        print(Fore.GREEN +"test: loss_s: {} loss_c: {}".format(loss_s.item(), 
                loss_c.item()) + Style.RESET_ALL)

        TP = confusion_matrix_class.diag()[1].item()
        TN = confusion_matrix_class.diag()[0].item()
        FP = confusion_matrix_class[0,1].item()
        FN = confusion_matrix_class[1,0].item()
        print('TP:{} TN:{} FP:{} FN:{}'.format(TP,TN,FP,FN))

        precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true=torch.cat(labels), y_pred=torch.cat(preds))

        print(Fore.RED + 'class Accuracy: {}/{}({:.2f}%)'.format(correct,len(dataloader.dataset), 
                100 * correct / len(dataloader.dataset)) +Style.RESET_ALL)
        print(Fore.RED + 'Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}'.format(precision[0], 
                recall[0], fscore[0]) + Style.RESET_ALL)

        globalDice = 2 * np.sum(TP_sum) / (2 * np.sum(TP_sum) + np.sum(FP_sum) + np.sum(FN_sum))
        globalIOU = np.sum(TP_sum) / (np.sum(TP_sum) + np.sum(FP_sum) + np.sum(FN_sum))
        print(Fore.RED+"globalIOU: {}".format(globalIOU)+Style.RESET_ALL)
        print(Fore.RED+"globalDice: {}".format(globalDice)+Style.RESET_ALL)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--network", default="FPSNet", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--test_list", default="../sc_rate", type=str)
    parser.add_argument("--test_path", default="./saved/WIML_FPSNet_five_strong_labels.pth", type=str)
    parser.add_argument("--data_root", default="../test_data", type=str)
    parser.add_argument("--imagesave_dir", default="./saved/image_prediction", type=str)
   
    args = parser.parse_args()


    cudnn.benchmark = True

    ## Data loader
    test_dataset = TestingDataset(args.test_list, args.data_root,
                                transform_c=x_transform, target_transform_s =y_transform)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                shuffle=False, num_workers=args.num_workers)
    
    # Supervised loss
    criterion_s = torch.nn.BCEWithLogitsLoss()
    criterion_c = torch.nn.CrossEntropyLoss()

    ## Load model
    model = network.FPSNet()
    checkpoint = torch.load(args.test_path)
    model.load_state_dict(checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model.to(device)
    
    tester(model, criterion_s, criterion_c, test_dataloader, device, args.imagesave_dir)