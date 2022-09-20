import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from skimage import color, measure

def show_heatmap(heatmap1, heatmap2):
    heatmap1 = heatmap1.detach().data.cpu().numpy()[0]
    heatmap1 = np.mean(heatmap1, axis=0)
    heatmap2 = heatmap2.detach().data.cpu().numpy()[0]
    heatmap2 = np.mean(heatmap2, axis=0)
    plt.subplot(121)
    plt.imshow(heatmap1)
    plt.subplot(122)
    plt.imshow(heatmap2)
    plt.show()

def index_choice(label_seg, num_vec=1):
    label_seg_np = label_seg.cpu().data.numpy()
    batch_size = label_seg_np.shape[0]
    label_positive_index = np.argwhere(label_seg_np != 0)
    label_negtive_index = np.argwhere(label_seg_np == 0)
    label_positive_num = label_positive_index.shape[0]
    label_negtive_num = label_negtive_index.shape[0]
    if num_vec * batch_size > label_positive_num:
        label_chosen_num = label_positive_num
    else:
        label_chosen_num = num_vec * batch_size
    label_positive_index_index = np.random.choice(np.arange(label_positive_num), label_chosen_num, replace=False)
    label_negtive_index_index = np.random.choice(np.arange(label_negtive_num),label_chosen_num, replace=False)
    label_positive_selected_index = label_positive_index[label_positive_index_index]
    label_negtive_selected_index = label_negtive_index[label_negtive_index_index]
    
    return label_positive_selected_index, label_negtive_selected_index

def compute_triplet_loss(feature, word_f, label_seg, criterion_t):
    
    word_f_mod = torch.norm(word_f).item()
    word_f_norm = word_f / word_f_mod
    label_seg = label_seg.clone()
    label_seg_max = np.max(label_seg.cpu().numpy())
    if label_seg_max !=0:
        pos_index_np, neg_index_np = index_choice(label_seg, num_vec=4)
        pos_index = torch.tensor(pos_index_np)
        neg_index = torch.tensor(neg_index_np)
        loss_total_t = 0
        count = 0
        for num in range(pos_index.shape[0]):
            pos_index_item = pos_index[num]
            neg_index_item = neg_index[num]
            positive = feature[pos_index_item[0], :, pos_index_item[2], pos_index_item[3]].unsqueeze(0)
            negative = feature[neg_index_item[0], :, neg_index_item[2], neg_index_item[3]].unsqueeze(0)

            positive = positive / torch.norm(positive)
            negative = negative / torch.norm(negative)

            loss_t_f = criterion_t(word_f_norm.view(1, -1), positive, negative)

            loss_total_t = loss_total_t + loss_t_f
            count = count + 1
        
    else:
        loss_total_t = torch.tensor(0.)
        count = 1
    
    return loss_total_t / count

def compute_infonce(feature, word_f, label_seg):
    word_f_mod = torch.norm(word_f).item()
    word_f_norm = word_f / word_f_mod
    label_seg = label_seg.clone()
    label_seg_max = np.max(label_seg.cpu().numpy())
    if label_seg_max !=0:
        pos_index_np, neg_index_np = index_choice(label_seg, num_vec=4)
        pos_index = torch.tensor(pos_index_np)
        neg_index = torch.tensor(neg_index_np)
        loss_total_t = 0
        count = 0

        cosine_sim_list = []
        pos_index_item = pos_index[0]
        positive = feature[pos_index_item[0], :, pos_index_item[2], pos_index_item[3]].unsqueeze(0)
        cosine_sim_pos = F.cosine_similarity(positive, word_f.view(1, -1))
        cosine_sim_list.append(cosine_sim_pos)
        for num in range(neg_index.shape[0]):
            neg_index_item = neg_index[num]
            negative = feature[neg_index_item[0], :, neg_index_item[2], neg_index_item[3]].unsqueeze(0)

            cosine_sim_neg = F.cosine_similarity(negative, word_f.view(1, -1))
            cosine_sim_list.append(cosine_sim_neg)
        cosine_sim_tensor = torch.cat(cosine_sim_list, dim=0)
        cosine_sim_tensor = cosine_sim_tensor / 0.07
        cosine_sim_log_softmax = F.log_softmax(cosine_sim_tensor)
        loss_infonce = -cosine_sim_log_softmax[0]
        
    else:
        loss_infonce = torch.tensor(0.)
    
    return loss_infonce
 
def ROC_PR(scores, labels):

    score_tensor = torch.cat(scores)
    score = np.array(score_tensor)
    print(score)
    label_tensor = torch.cat(labels)
    label = np.array(label_tensor)
    print(label)

    fpr, tpr, thresholds  =  roc_curve(label, score)
    auc1 = auc(fpr, tpr)
    print('AUC:'+str(auc1))

    precision, recall, thresholds = precision_recall_curve(label,score) 
    # print(thresholds)
    plt.subplot(121)
    plt.title('PR')
    plt.plot(recall,precision)

    plt.subplot(122)
    plt.title('ROC')
    plt.plot(fpr,tpr)

    plt.show()


def largestConnectComponent(img):
        binaryimg = img

        label_image, num = measure.label(
            binaryimg, background=0, return_num=True) 
        areas = [r.area for r in measure.regionprops(label_image)] 
        areas.sort()
       
        if len(areas) > 1:
            for region in measure.regionprops(label_image):
                if region.area < areas[-1]:
                    for coordinates in region.coords: #coords 
                        label_image[coordinates[0], coordinates[1]] = 0
        label_image = label_image.astype(np.int8)
        label_image[np.where(label_image > 0)] = 1
        return label_image