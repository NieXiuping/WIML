from torch.nn.functional import pairwise_distance
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np
# np.random.seed(0)

class TrainingDataset(Dataset):
    def __init__(self, root_list, root_data, 
                 transform_s=None, target_transform_s=None, transform_c=None): 

        fh_seg = open(os.path.join(root_list, 'train_seg_list.txt'), 'r')
        label_seg_list = []
        mask_seg_list = []
        for line in fh_seg:
            line = line.rstrip()
            words = line.split()
            img = os.path.join(root_data, 'data', words[0]) 
            mask = os.path.join(root_data, 'label',words[0])
            label_seg_list.append((words[0], int(words[1])))
            mask_seg_list.append([img, mask])

        fh_cls = open(os.path.join(root_list, 'train_cls_list.txt'), 'r')
        label_cls_list = []
        for line in fh_cls:
            line = line.rstrip()
            words = line.split()
            label_cls_list.append((words[0], int(words[1])))
        
        weights_seg = self.get_weights_for_balanced_classes(label_seg_list, 2)
        weights_cls = self.get_weights_for_balanced_classes(label_cls_list, 2)
        self.prob_seg = np.array(weights_seg) / sum(weights_seg)
        self.prob_cls = np.array(weights_cls) / sum(weights_cls)

        self.root_list = root_list
        self.root_data = root_data
        self.label_seg_list = label_seg_list
        self.mask_seg_list = mask_seg_list
        self.label_cls_list = label_cls_list
        self.transform_s = transform_s
        self.target_transform_s = target_transform_s
        self.transform_c = transform_c

    def __len__(self): 
        return len(self.label_cls_list)

    def get_weights_for_balanced_classes(self, labels, nclasses):
        count = [0] * nclasses
        for item in labels:
            count[item[1]] += 1
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            if count[i] != 0:
                weight_per_class[i] = N / float(count[i])
        weight = [0] * len(labels)
        for idx, val in enumerate(labels):
            weight[idx] = weight_per_class[val[1]]
        return weight
        
    def __getitem__(self, index):   

        fn_num_seg = len(self.mask_seg_list)
        fn_index_seg = np.random.choice(np.arange(fn_num_seg), 1, replace=True, p=self.prob_seg)
        x_path, y_path = self.mask_seg_list[fn_index_seg.item()] 
        img_x = Image.open(x_path)
        img_x = np.asarray(img_x.convert('RGB'))
        img_y = np.asarray(Image.open(y_path))

        if self.transform_s is not None:
            img_x = self.transform_s(img_x)
        if self.target_transform_s is not None:
            img_y = self.target_transform_s(img_y)

        fn_num_cls = len(self.label_cls_list)
        fn_index_cls = np.random.choice(np.arange(fn_num_cls), 1, replace=True, p=self.prob_cls)
        fn_cls, label_cls = self.label_cls_list[fn_index_cls.item()] 
        img_path_cls = os.path.join(self.root_data, 'data', fn_cls)
        frame_cls = Image.open(img_path_cls)
        frame_cls = frame_cls.convert('RGB')

        if self.transform_c is not None:
            frame_cls = self.transform_c(frame_cls)
        label_cls = torch.tensor(label_cls)
        return img_x, img_y, frame_cls, label_cls 


class TestingDataset(Dataset):
    def __init__(self, root_list, root_data,
                    transform_c=None, target_transform_s=None): 
        imgs = []
        fh = open(os.path.join(root_list, 'test.txt'), 'r')
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))

        self.root_list = root_list
        self.root_data = root_data
        self.imgs = imgs
        self.transform_c = transform_c
        self.target_transform_s = target_transform_s

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):   

        fn, label = self.imgs[index] 
        img_path=os.path.join(self.root_data, 'data', fn)
        frame=Image.open(img_path)
        frame = frame.convert('RGB')

        if self.transform_c is not None:
            frame = self.transform_c(frame)
        label=torch.tensor(label)

        y_path=os.path.join(self.root_data, 'label', fn)
        img_y = Image.open(y_path)

        if self.target_transform_s is not None:
            img_y = self.target_transform_s(img_y) 
        
        return frame, label, img_y, fn

