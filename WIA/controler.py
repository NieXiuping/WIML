
import os
from collections import OrderedDict
from os.path import join as opj
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import transforms as T
import maxflow
import numpy as np 
import torch
from PIL import Image
from scipy import ndimage
import time
from scipy.ndimage import zoom
from colorama import Fore, Back, Style 
from skimage import color, measure
import network
from utils import (add_countor, add_overlay, cropped_image, extends_points,
                   extreme_points, get_bbox, get_largest_two_component,
                   get_start_end_points, interaction_euclidean_distance,
                   interaction_gaussian_distance,
                   interaction_geodesic_distance,
                   interaction_refined_geodesic_distance,
                   itensity_normalization, itensity_normalize_one_volume,
                   itensity_standardization, softmax, softmax_seg, zoom_image,
                   img_transform, img_preprocess, comp_class_vec)

rootPATH = os.path.abspath(".")

class Controler(object):
    seeds = 0
    extreme_points = 5
    foreground = 2
    background = 3
    #position
    imageName = "./logo.png"
    model_path = os.path.join("./weakly_supervised_model.pth")
    root_label = "./label"
    root_result = "result/initial_mask"
    root_cam = "result/CAM"

    fmap_block = list()
    grad_block = list()

    def __init__(self):
        self.img = None
        self.step = 0
        self.image = None
        self.mask = None
        self.overlay = None
        self.seed_overlay = None
        self.segment_overlay = None
        self.extreme_point_seed = []
        self.background_seeds = []
        self.foreground_seeds = []
        self.current_overlay = self.seeds
        self.load_image(self.imageName)
        self.input_img = None
        self.initial_seg = None
        self.initial_extreme_seed = None
        self.cam_img = None
        self.start_time = 0
        self.end_time = 0
        self.total_time = 0
        self.refine_iou = 0
        self.refine_dice = 0
        self.total_num = 0
        self.init_iou = 0
        self.init_dice = 0

    def initial_param(self):
        self.step = 0
        self.img = None
        self.image = None
        self.mask = None
        self.overlay = None
        self.seed_overlay = None
        self.segment_overlay = None
        self.extreme_point_seed = []
        self.background_seeds = []
        self.foreground_seeds = []
        self.current_overlay = self.seeds
        self.initial_seg = None
        self.initial_extreme_seed = None
        self.input_img = None
        self.cam_img = None
        self.start_time = 0
        self.end_time = 0
        self.total_time = 0
        self.refine_iou = 0
        self.refine_dice = 0
        self.total_num = 0
        self.init_iou = 0
        self.init_dice = 0

    def load_image(self, filename):
        self.filename = filename
        self.initial_param() #re-initial

        self.init_image = zoom_image(cv2.imread(filename), num=3)#display the image after "clear"
        self.image = zoom_image(cv2.imread(filename), num=3) #display the image
        self.img = zoom_image(np.array(Image.open(filename).convert('L')), num=2) #gray image 
        self.images = zoom_image(cv2.imread(filename), num=3) #refine 
        self.input_img = zoom_image(np.array(Image.open(filename).convert('RGB')), num=3)
        self.cam_img = cv2.imread(filename)

        self.seed_overlay = np.zeros_like(self.image)
        self.segment_overlay = np.zeros_like(self.image)
        self.mask = None
        self.refined_clicks = 0
        self.refined_iterations = 0

    def add_seed(self, x, y, type):
        if self.image is None:
            print('Please load an image before adding seeds.')
        if type == self.background:
            if not self.background_seeds.__contains__((x, y)): 
                self.background_seeds.append((x, y))
                cv2.rectangle(self.seed_overlay, (x - 1, y - 1),
                              (x + 1, y + 1), (255, 0, 255), 2) 
        
        elif type == self.foreground:
            if not self.foreground_seeds.__contains__((x, y)):
                if self.step == 0:
                    self.extreme_point_seed.append((x, y))
                    cv2.rectangle(self.seed_overlay, (x - 1, y - 1),
                                  (x + 1, y + 1), (255, 255, 0), 2)
                if self.step == 1:
                    self.foreground_seeds.append((x, y))
                    cv2.rectangle(self.seed_overlay, (x - 1, y - 1),
                                  (x + 1, y + 1), (0, 0, 255), 2)
                if len(self.extreme_point_seed) == 1:
                    import time
                    self.stage1_begin = time.time()
        if len(self.background_seeds) > 0 or len(self.foreground_seeds) > 0:
            self.refined_clicks += 1

        if self.refined_clicks == 1:
            import time
            self.start_time = time.time()
        if self.refined_clicks == 0:
            import time
            self.stage2_begin = None

    def clear_seeds(self):
        self.step = 0
        self.background_seeds = []
        self.foreground_seeds = []
        self.extreme_point_seed = []
        self.background_superseeds = []
        self.foreground_superseeds = []
        self.seed_overlay = np.zeros_like(self.seed_overlay)
        self.image = self.init_image
        self.start_time = 0
        self.end_time = 0
        self.total_time = 0

    def get_image_with_overlay(self, overlayNumber):
        return cv2.addWeighted(self.image, 0.9, self.seed_overlay, 0.7, 0.7) #dst = src1 * alpha + src2 * beta + gamma;

    def segment_show(self):
        pass

    def save_image(self):
        H, W, C = self.cam_img.shape
        if self.mask is None:
            print('Please segment the image before saving.')
            return
        self.mask = self.mask * 255
        self.mask = cv2.resize(self.mask, (512, 512))
        cv2.imwrite(str(self.filename.replace('data_source','result/strong_label')), self.mask.astype(int))
    
        print('image_name {} total_time {:.4f}s total_num {}'.format(self.filename.split("/")[-1], self.total_time, self.total_num))

    def compute_metrics(self, output, target):
        
        output = output.copy().astype(np.int)
        target = target.copy().astype(np.int)
        IOU = (output*target).sum()/((output | target).sum())

        Dice = 2*(output*target).sum()/(output.sum()+target.sum())

        return IOU, Dice
    
    def show_heatmap(self, heatmap):
    
        # heatmap = np.mean(heatmap, axis=0)
        plt.imshow(heatmap)
        plt.show()
        # plt.savefig("./image_result_finetune/{}.png".format(item))
        # plt.clf()

    def largestConnectComponent(self, img):
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

    def refined_seg(self):

        path_img = self.filename.split("/")[-2]
        output_dir = self.filename.replace(path_img, self.root_result)
        label_dir = self.filename.replace(path_img, self.root_label)
        label = Image.open(label_dir)
        label = np.array(label)
        label[label == 255] = 1
        label = cv2.resize(label, (256,256))

        fore_seeds = np.zeros_like(self.img)
        for i in self.foreground_seeds:
            fore_seeds[i[1], i[0]] = 1
        back_seeds = np.zeros_like(self.img)
        for i1 in self.background_seeds:
            back_seeds[i1[1], i1[0]] = 1

        fore_seeds = extends_points(fore_seeds)
        back_seeds = extends_points(back_seeds)

        normal_img = itensity_standardization(self.img)
        init_seg = [self.initial_seg, 1.0-self.initial_seg] #initial probability map(background and foreground)

        fg_prob = init_seg[0]
        bg_prob = init_seg[1]

        fore_geos = interaction_refined_geodesic_distance(
            normal_img, fore_seeds)     
        back_geos = interaction_refined_geodesic_distance(
            normal_img, back_seeds)

        # imporve the probability of fore and back 
        fore_prob = np.maximum(fore_geos, fg_prob)
        back_prob = np.maximum(back_geos, bg_prob)

        crf_seeds = np.zeros_like(fore_seeds, np.uint8)
        crf_seeds[fore_seeds > 0] = 170 
        crf_seeds[back_seeds > 0] = 255
        crf_param = (5.0, 0.1)

        crf_seeds = np.asarray([crf_seeds == 255, crf_seeds == 170], np.uint8)
        crf_seeds = np.transpose(crf_seeds, [1, 2, 0])

        x, y = fore_prob.shape
        prob_feature = np.zeros((2, x, y), dtype=np.float32)
        prob_feature[0] = fore_prob
        prob_feature[1] = back_prob
       
        softmax_feture = np.exp(prob_feature) / \
            np.sum(np.exp(prob_feature), axis=0)
        softmax_feture = np.exp(softmax_feture) / \
            np.sum(np.exp(softmax_feture), axis=0)

        fg_prob = softmax_feture[0].astype(np.float32)
        bg_prob = softmax_feture[1].astype(np.float32)

        Prob = np.asarray([bg_prob, fg_prob])
        Prob = np.transpose(Prob, [1, 2, 0])

        refined_pred = maxflow.interactive_maxflow2d(
            normal_img, Prob, crf_seeds, crf_param)

        pred = refined_pred.copy()

        pred = self.largestConnectComponent(pred)
        strt = ndimage.generate_binary_structure(2, 1)
        seg = np.asarray(
            ndimage.morphology.binary_opening(pred, strt), np.uint8)
        seg = np.asarray(
            ndimage.morphology.binary_closing(seg, strt), np.uint8)
        seg = ndimage.binary_fill_holes(seg)
        seg = np.clip(seg, 0, 255) 
        seg = np.array(seg, np.uint8) 

        self.refine_iou, self.refine_dice = self.compute_metrics(seg, label)

        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time 
        foreground_seeds_num = len(self.foreground_seeds)
        background_seeds_num = len(self.background_seeds)
        self.total_num = foreground_seeds_num + background_seeds_num
      
        with open('./result/recording_txt/{}.txt'.format(self.filename.split("/")[-1]),'w') as f:
            f.write('image_name {} cam_init_segmentation IOU {:.4f} DICE {:.4f} refine_segmentation IOU {} DICE {} total_time {:.4f}s total_num {}'.format(self.filename.split("/")[-1], self.init_iou, self.init_dice, self.refine_iou, self.refine_dice, self.total_time, self.total_num)+'\n')
        
        print("refine_segmentation IOU {:.4f} DICE {:.4f}".format(self.refine_iou, self.refine_dice)) 
              
        img = self.images.copy()
        contours_label, hierarchy = cv2.findContours(
            label.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:] 
        if len(contours_label) != 0:
            image_label= cv2.drawContours(
                self.images, contours_label, -1, (255, 0, 0), 1)

        contours, hierarchy = cv2.findContours(
            seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:] 
        if len(contours) != 0:
            image_data = cv2.drawContours(
                image_label, contours, -1, (0, 255, 0), 1)
        
        self.images = img
        self.image = image_data
        self.mask = seg

        # refine_seg = seg.copy()
        # refine_seg = cv2.resize(refine_seg* 255, (512, 512))
        # cv2.imwrite(output_dir, refine_seg.astype(int))

#-------------------------GradCAM------------------------#
    def backward_hook(self, module, grad_in, grad_out):
        self.grad_block.append(grad_out[0].detach())

    def farward_hook(self, module, input, output):
        self.fmap_block.append(output)  

    def cam_show_img(self, img, feature_map, grads):
        H, W, _ = img.shape
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		
        grads = grads.reshape([grads.shape[0],-1])					
        weights = np.mean(grads, axis=1)							
        for i, w in enumerate(weights):
            cam += w * feature_map[i, :, :]							
        cam = np.maximum(cam, 0)

        cam = cam / cam.max()
        cam = cv2.resize(cam, (W, H))

        #############
        cam2 = cam.copy()
        cam2[cam2>=0.78]=0.78
        cam2 = cam2 / cam2.max()
        cam2 = cv2.resize(cam2, (W, H))
        #############
        heatmap = cv2.applyColorMap(np.uint8(255 * cam2), cv2.COLORMAP_JET)
        cam_img = 0.3 * heatmap + 0.7 * img

        cam_init = cam.copy()
        init_seg = [cam_init, 1.0-cam_init]
        fg_prob = init_seg[0] 
        bg_prob = init_seg[1] 
        crf_param = (5.0, 0.1) 
        Prob = np.asarray([bg_prob, fg_prob])
        Prob = np.transpose(Prob, [1, 2, 0]) # H W C
      
        fix_predict = maxflow.maxflow2d(img.astype(
            np.float32), Prob, crf_param) 
        fixed_predict = zoom(fix_predict, (1, 1), output=None,
                             order=0, mode='constant', cval=0.0, prefilter=True)
        
        pred = fixed_predict.copy()
       
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        pred = self.largestConnectComponent(pred)
        strt = ndimage.generate_binary_structure(2, 1)
        seg = np.asarray(
            ndimage.morphology.binary_opening(pred, strt), np.uint8)
        seg = np.asarray(
            ndimage.morphology.binary_closing(seg, strt), np.uint8)
        seg = ndimage.binary_fill_holes(seg)
        seg = np.clip(seg, 0, 255) 
        seg = np.array(seg, np.uint8)

        self.initial_seg = cam_init  
        # cv2.imwrite(out_dir, 255 *seg)
        return seg, cam_img

    def cam_segmentation(self):
        if self.step == 0:
            
            path_img = self.filename.split("/")[-2]
            output_dir = self.filename.replace(path_img, self.root_result)
            cam_dir = self.filename.replace(path_img, self.root_cam)
            label_dir = self.filename.replace(path_img, self.root_label)
            label = Image.open(label_dir)
            label = np.array(label)
            label[label == 255] = 1
            label = cv2.resize(label, (256,256))

            classes = ('0', '1') 
            self.fmap_block = list()
            self.grad_block = list()

            # network loading
            img_input = img_preprocess(self.cam_img)

            model = network.FPSNet()
            model.load_state_dict(torch.load(self.model_path))
            model.eval()

            # registrate hook
            model.net.outconv.register_forward_hook(self.farward_hook)	
            model.net.outconv.register_backward_hook(self.backward_hook)

            # forward
            _, output = model(img_input)
       
            idx = np.argmax(output.cpu().data.numpy())
            print("predict: {}".format(classes[idx]))

            # backward
            model.zero_grad()
            class_loss = comp_class_vec(output)
            # print(class_loss)
            class_loss.backward()

            # generate cam
            grads_val = self.grad_block[0].cpu().data.numpy().squeeze()
            fmap = self.fmap_block[0].cpu().data.numpy().squeeze()

            # save cam
            img_show = np.float32(cv2.resize(self.cam_img, (256, 256)))
            seg, cam_img = self.cam_show_img(img_show, fmap, grads_val)
            cv2.imwrite(cam_dir, cam_img)
            seg2 = cv2.resize(seg.copy(), (512,512))
            cv2.imwrite(output_dir, 255*seg2)

            self.init_iou, self.init_dice = self.compute_metrics(seg, label)
            print(Fore.GREEN+'image_name:{}'.format(self.filename.split("/")[-1])+Style.RESET_ALL)
            print("cam_init_segmentation IOU {:.4f} DICE {:.4f}".format(self.init_iou, self.init_dice))

            #Draw pred contours
            contours_label, hierarchy = cv2.findContours(
                label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:] 
            if len(contours_label) != 0:
                image_label = cv2.drawContours(
                    self.image, contours_label, -1, (255, 0, 0), 1)

            contours, hierarchy = cv2.findContours(
                seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:] 
            if len(contours) != 0:
                image_data = cv2.drawContours(
                    image_label, contours, -1, (0, 255, 0), 1)

            self.image = image_data
            self.mask = seg
            self.step = 1 #step=1 initial seg; step=2 refine seg 




