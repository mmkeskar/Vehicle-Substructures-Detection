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
        
        self.keypoint_ids = [0, 1, 2, 3, 22, 23, 25, 26, 31, 32, 34, 35, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
        
        
        
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
        
        ann_id = self.coco.getAnnIds([image_id])
        annotations = self.coco.loadAnns(ann_id)
        
        labels = []
        
        """bboxes = []
        keypoints = []"""
        for ann in annotations:
            if len(ann['keypoints']) != 0:
                label = {}
                label['bbox'] = np.array((ann['bbox']))
                label['keypoints'] = []
                for key_id in self.keypoint_ids:
                    label['keypoints'].append([ann['keypoints'][3*key_id], ann['keypoints'][3*key_id + 1], ann['keypoints'][3*key_id + 2]])
                label['keypoints'] = np.array(label['keypoints'])
                labels.append(label)  
        """bboxes = np.array(bboxes)   
        keypoints = np.array(keypoints)
        
        if bboxes.shape[0] < self.max_objects:
            padding = -np.ones((self.max_objects-bboxes.shape[0], bboxes.shape[1]))
            bboxes = np.vstack((bboxes, padding))
            
        if keypoints.shape[0] < self.max_objects*24:
            padding = -np.ones((self.max_objects*24-keypoints.shape[0], keypoints.shape[1]))
            keypoints = np.vstack((keypoints, padding))"""
        
        sample = {'image': image, 'image_path': image_path, 'labels': labels}
        #print(f"bounding boxes matrix size: {bboxes.shape}")
        #print(f"keypoints matrix size: {keypoints.shape}")
        return sample
    
    def __len__(self):
       return self.sample_size