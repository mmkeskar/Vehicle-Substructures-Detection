#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:54:11 2021
@author: akshay
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os

from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg

"""from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg"""
import math

class VehIntDataset(data.Dataset):
    
    def __init__(self, data_dir, opt):
        super(VehIntDataset, self).__init__()
        self.data_dir = data_dir
        self.opt = opt
        
        self.image_path_const = "/media/akshay/SSD1/data-apollocar3d/train/images"
        self.coco = coco.COCO(self.data_dir)
        self.images = self.coco.getImgIds()
        
        self.sample_size = len(self.images)
        
        self.max_objects = 49 # found using python script
        
        self.keypoint_ids = [0, 1, 2, 3, 22, 23, 25, 26, 31, 32, 34, 35, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
        self.num_kpts = len(self.keypoint_ids)
        
        
        
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
            return border // i


    def __getitem__(self, index):
        image_id = self.images[index]
        file_path_ext = self.coco.loadImgs(image_id)[0]['file_name']
        image_path = os.path.join(self.image_path_const, file_path_ext)
        print(f"Image path is: {image_path}")
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
        output_res = self.opt.output_res
        
        ann_id = self.coco.getAnnIds([image_id])
        annotations = self.coco.loadAnns(ann_id)
        
        # phi function that gives the heatmap for every keypoint
        kpt_hm = np.zeros((self.num_kpts, image_size[0], image_size[1]), dtype=np.float32)
        # J function that gives the offset from to the keypoints from the center of the object

        # a heatmap for the object centers
        hm = np.zeros((self.num_cats, output_res, output_res), dtype=np.float32)
        # a heatmap for all the keypoints
        hm_hp = np.zeros((self.max_kpts, output_res, output_res), dtype=np.float32)
        # the offsets of the keypoints from the object center points
        kps = np.zeros((self.max_objects, self.max_kpts*2), dtype=np.float)
        reg_mask = np.zeros((self.max_objects), dtype=np.unit8)
        kps_mask = np.zeros((self.max_objects, self.max_kpts*2), dtype=np.unit8)
        wh = np.zeros((self.max_objects, 2), dtype=np.float32)
        hp_mask = np.zeros((self.max_objects*self.max_kpts), dtype=np.int64)
        reg = np.zeros((self.max_objects, 2))
        hp_offset = np.zeros((self.max_objects*self.max_kpts, 2), dtype=np.float32)
        ind = np.zeros((self.max_objects), dtype=np.int64)
        hp_ind = np.zeros((self.max_objects*self.max_kpts), dtype=np.int64)
        
        dense_kps = np.zeros((self.max_kpts, 2, output_res, output_res), dtype=np.float32)
        dense_kps_mask = np.zeros((self.max_kpts, output_res, output_res), dtype=np.float32)
        
        gt_det = []
        
        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian
        
        print("creating targets")
        for i in range(num_objs):
            ann = annotations[i]
            bbox = ann['bbox']
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            center = np.array([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
            ct_int = np.array([int(center[0]), int(center[1])])
            
            ind[i] = ct_int[1]*output_res + ct_int[0]
            reg[i] = center - ct_int
            
            wh[i] = w, h
            
            radius = gaussian_radius((h, w))
            sigma = radius/3
            
            
            if len(ann['keypoints']) != 0:
                kpts = np.array(ann['keypoints']).reshape(self.num_kpts, 3)
                reg_mask[i] = 1
                
                for j in range(self.max_kpts):
                    kpt = kpts[self.keypoint_ids[j]]
                    
                    hp_offset[i*self.max_kpts+j] = kpt[:2] - kpt[:2].astype(np.int32)
                    
                    draw_gaussian(hm_hp[j], (kpt[0], kpt[1]), radius)
                    
                    kps[i][j*2:j*2+2] = kpt[0] - center[0], kpt[1] - center[1]
                    
                    hp_mask[i*self.max_keypoints+j] = 1
                    
                    kps_mask[i][j*2:j*2+2] = 1
                    
                    hp_ind[i*self.max_kpts + j] = int(kpt[1])*output_res + int(kpt[0])
                    
                    if self.opt.dense_hp:
                        draw_dense_reg(dense_kps, hm[0], ct_int, kps[i][j*2:j*2+2], radius, True)
                        draw_gaussian(dense_kps_mask, ct_int, radius)
                
                gt_det.append([center[0] - w/2, center[1] - h/2,
                           center[0] + w/2, center[1] + h/2, 1] + kpts[:, :2].reshape[self.max_kpts*2].tolist() + [0])
           
            draw_gaussian(hm[0], ct_int, sigma)
            
        sample = {'image': image, 'image_path': image_path, 'hm': hm, 'wh': wh, 'hp_mask': hp_mask, \
                  'reg': reg, 'kps': kps, 'hm_hp': hm_hp, 'reg_mask': reg_mask, \
                  'kps_mask': kps_mask, 'gt_det': gt_det, 'ind': ind, 'hp_ind': hp_ind} 
            
        return sample
        
        
    def __len__(self):
       return self.sample_size
