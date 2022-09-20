from numpy.core.defchararray import count
import torch
from PIL import Image
from torchvision.transforms import transforms as T
import argparse 
import network.network_FPSNet as network
from dataloader.dataset import TrainingDataset
from torch import optim
from numpy import random
from torch.utils.data import DataLoader, dataloader
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn 
import torch.nn.functional as F
import torch.nn as nn
import cv2
import numpy as np
import colorama 
from colorama import Fore, Back, Style
import os
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
from sklearn.metrics import precision_recall_fscore_support 
from tensorboardX import SummaryWriter
from warnings import simplefilter
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=DeprecationWarning)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#Input image transform
x_transform = T.Compose([
    T.ToTensor(),
    T.Resize(size=(160,160)), 
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
#Label transform
y_transform = T.Compose([
    T.ToTensor(),
    T.Resize(size=(160,160))
])

def trainer(model, criterion_s, criterion_c, optimizer, device, save_root,
            dataloader, epoch, max_epoches, writer):
    
    print("Epoch {} Train Model".format(epoch).center(60, '-'))
    print('Epoch {}/{}'.format(epoch, max_epoches - 1))
    print('-' * 10)
    dataset_size = len(dataloader.dataset)
    batch_num = dataset_size // dataloader.batch_size
    item_num_train = epoch*batch_num
    epoch_loss = 0
    step = 0 

    model.train()
    
    for x_s, y_s, x_c, y_c in dataloader:

        ## Input data
        inputs_s = x_s.to(device)
        labels_s = y_s.to(device)
        outputs_s, _ = model(inputs_s)
        inputs_c = x_c.to(device)
        labels_c = y_c.to(device)
        _, outputs_c = model(inputs_c)
        outputs_c = outputs_c.squeeze()

        ## Compute loss
        loss_c = criterion_c(outputs_c, labels_c) # segmentation loss
        loss_s = criterion_s(outputs_s, labels_s) # classification loss
        loss_total = loss_c + loss_s

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        epoch_loss += loss_total.item()
        step += 1
        item_num_train = item_num_train + step

        writer.add_scalars('train_loss_s', {'loss_s': loss_s}, global_step=item_num_train)
        writer.add_scalars('train_s', {'loss_s': loss_s}, global_step=epoch)
        writer.add_scalars('train_loss_c', {'loss_c': loss_c}, global_step=item_num_train)
        writer.add_scalars('train_c', {'loss_c': loss_c}, global_step=epoch)

        print(Fore.RED + "%d/%d,train_loss:%0.3f, loss_s: {} loss_c: {}" 
                % (step, batch_num, loss_total.item(), loss_s.item(), loss_c.item()) + Style.RESET_ALL)
        
    print(Fore.RED + "epoch %d avg_loss:%0.3f" % (epoch, epoch_loss // batch_num) + Style.RESET_ALL) #compute the avg_loss of this epoch

    ## Save the trained model
    torch.save(model.state_dict(), os.path.join(save_root, "weights_%d.pth" % epoch))

    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_epoches", default=60, type=int)
    parser.add_argument("--network", default="FPSNet", type=str)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=1e-4, type=float)
    parser.add_argument("--train_list", default="../sc_rate", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--pretrained_weights", default="./model/best_model.pth", type=str)
    parser.add_argument("--data_root", default="../data_source", type=str)
    parser.add_argument("--save_dir", default="./saved", type=str)
   
    args = parser.parse_args()

    cudnn.benchmark = True
    cudnn.deterministic = False
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    ## Data loader
    train_dataset = TrainingDataset(args.train_list, args.data_root,
                                    transform_s=x_transform, target_transform_s=y_transform,
                                    transform_c=x_transform)
    
    train_dataloader= DataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=args.num_workers, 
                                drop_last=True)

    ## Supervised loss
    criterion_s = torch.nn.BCEWithLogitsLoss()
    criterion_c = torch.nn.CrossEntropyLoss()

    ## Load model
    model = network.FPSNet()
    # load the pretrained model
    pretrain_path = args.pretrained_weights
    checkpoint = torch.load(pretrain_path)
    model_dict = model.state_dict()
    new_dict = {k: v for k,v in checkpoint.items() if k in model_dict}
    print(new_dict.keys())
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wt_dec)
    writer = SummaryWriter()
   
    for epoch in range(args.max_epoches):
        trainer(model, criterion_s, criterion_c, optimizer, device, args.save_root,
                train_dataloader, epoch, args.max_epoches, writer)
