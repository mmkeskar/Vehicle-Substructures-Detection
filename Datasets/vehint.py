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

from helper_functions import gaussian_heatmap, get_radius
"""from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg"""
import math

class VehIntDataset(data.Dataset):
    
    def __init__(self, data_dir):
        super(VehIntDataset, self).__init__()
        self.data_dir = data_dir
        self.image_path_const = "/media/akshay/SSD1/data-apollocar3d/train/images"
        self.coco = coco.COCO(self.data_dir)
        self.images = self.coco.getImgIds()
        
        self.sample_size = len(self.images)
        
        self.max_objects = 49 # found using python script
        self.num_kpts = 66
        self.num_cats = 1
        self.keypoint_ids = [0, 1, 2, 3, 22, 23, 25, 26, 31, 32, 34, 35, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
        self.max_kpts = len(self.keypoint_ids)
        
        
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
        
        image_size = (image.shape[1], image.shape[2])
        
        ann_id = self.coco.getAnnIds([image_id])
        annotations = self.coco.loadAnns(ann_id)
        
        num_objs = min(self.max_objects, len(annotations))
        
        # a heatmap for the object centers
        obj_hm = np.zeros((self.num_cats, image_size[0], image_size[1]), dtype=np.float32)
        # a heatmap for all the keypoints
        kpt_hm = np.zeros((self.max_kpts, image_size[0], image_size[1]), dtype=np.float32)
        # the offsets of the keypoints from the object center points
        kpt_regs = np.zeros((self.max_objects, self.max_kpts, 2), dtype=np.float)
       
        ct_offset = np.zeros((self.max_objects, 2))
        
        for i in range(num_objs):
            ann = annotations[i]
            bbox = ann['bbox']
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            
            center = np.array([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
            ct_int = np.array([int(center[0]), int(center[1])])
            
            ct_offset[i] = center - ct_int
            
            radius = get_radius(bbox[2] - bbox[0], bbox[3] - bbox[1])
            sigma = radius/3
            
            obj_hm[0] = gaussian_heatmap(obj_hm[0], ct_int, sigma)
            
            if len(ann['keypoints']) != 0:
                kpts = np.array(ann['keypoints']).reshape(self.num_kpts, 3)
                for j in range(self.max_kpts):
                    kpt = kpts[self.keypoint_ids[j]]
                    
                    rad = get_radius(bbox[2] - bbox[0], bbox[3] - bbox[1])
                    
                    kpt_hm[j] = gaussian_heatmap(kpt_hm[j], (kpt[0], kpt[1]), rad)
                    
                    kpt_regs[i][j][0], kpt_regs[i][j][1] = kpt[0] - center[0], kpt[1] - center[1]
            
            
           
        sample = {'image': image, 'image_path': image_path, 'hm': obj_hm, \
                  'center offset': ct_offset, 'keypoint regression': kpt_regs, \
                  'keypoint heatmaps': kpt_hm} 
            
        return sample
    
    def __len__(self):
       return self.sample_size
