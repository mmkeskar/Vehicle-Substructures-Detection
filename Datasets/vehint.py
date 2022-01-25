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

from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
"""from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg"""
import math

class VehIntDataset(data.Dataset):

    def resize(self, arr, size):
        if size == 4:
            arr[0], arr[2] = arr[0] * (512 / 3384), arr[2] * (512 / 3384)
            arr[1], arr[3] = arr[1] * (512 / 2710), arr[3] * (512 / 2710)
            arr = arr / self.opt.down_ratio
        if size == 2:
            arr[0] = arr[0]*(512/3384)
            arr[1] = arr[1]*(512/2710)
            arr = arr/self.opt.down_ratio

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        #print("__get_item__ called")
        image_id = self.images[index]
        file_path_ext = self.coco.loadImgs(image_id)[0]['file_name']
        image_path = os.path.join(self.image_path_const, file_path_ext)
        #print(f"Image path is: {image_path}")
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        input_res = self.opt.input_res

        if self.split == 'train' and not self.opt.not_rand_crop:
            c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
            s = max(image.shape[0], image.shape[1]) * 1.0
            rot = 0

            s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
            w_border = self._get_border(128, image.shape[1])
            h_border = self._get_border(128, image.shape[0])
            c[0] = np.random.randint(low=w_border, high=image.shape[1] - w_border)
            c[1] = np.random.randint(low=h_border, high=image.shape[0] - h_border)

            trans_input = get_affine_transform(
                c, s, rot, [self.opt.input_res, self.opt.input_res])
            image = cv2.warpAffine(image, trans_input,
                                 (self.opt.input_res, self.opt.input_res),
                                 flags=cv2.INTER_LINEAR)
            image_to_show = image.copy()
            image = (image.astype(np.float32) / 255.)
            image = image.transpose(2, 0, 1)
        else:
            image = cv2.resize(image, (input_res, input_res), interpolation=cv2.INTER_AREA)
            image = (image.astype(np.float32) / 255.)
            image = image.transpose(2, 0, 1)
        
        output_res = self.opt.output_res
        
        ann_id = self.coco.getAnnIds([image_id])
        annotations = self.coco.loadAnns(ann_id)
        
        num_objs = min(self.max_objects, len(annotations))
        
        # a heatmap for the object centers
        hm = np.zeros((self.num_cats, output_res, output_res), dtype=np.float32)
        # a heatmap for all the keypoints
        hm_hp = np.zeros((self.max_kpts, output_res, output_res), dtype=np.float32)
        # the offsets of the keypoints from the object center points
        kps = np.zeros((self.max_objects, self.max_kpts*2), dtype=np.float)
        reg_mask = np.zeros((self.max_objects), dtype=np.uint8)
        kps_mask = np.zeros((self.max_objects, self.max_kpts*2), dtype=np.uint8)
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
        
        #print("creating targets")
        for i in range(num_objs):
            ann = annotations[i]
            bbox = ann['bbox']
            bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            if self.split == 'train' and not self.opt.not_rand_crop:
                trans_output = get_affine_transform(c, s, 0, [output_res, output_res])
                bbox[:2] = affine_transform(bbox[:2], trans_output)
                bbox[2:] = affine_transform(bbox[2:], trans_output)
                bbox = np.clip(bbox, 0, output_res - 1)
            else:
                self.resize(bbox, 4)
                bbox = np.clip(bbox, 0, output_res - 1)
            
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            center = np.array([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
            ct_int = np.floor(center).astype(np.int)
            
            ind[i] = ct_int[1]*output_res + ct_int[0]
            reg[i] = center - ct_int
            
            wh[i] = w, h
            
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            sigma = radius/3
            
            reg_mask[i] = 1

            image_to_show = cv2.circle(image_to_show, ct_int, 3, (0, 255, 0), 2)
            cv2.imshow("image", image_to_show)

            cv2.waitKey(0)
            
            
            if len(ann['keypoints']) != 0:
                kpts = np.array(ann['keypoints']).reshape(self.num_kpts, 3)
                kpts = kpts[self.keypoint_ids, :]
                if self.split == 'train' and not self.opt.not_rand_crop:
                    trans_output_rot = get_affine_transform(c, s, rot, [output_res, output_res])
                if self.opt.not_rand_crop:
                    self.resize(kpts[:, :2], 2)
                    kpts = np.clip(kpts, 0, output_res - 1)
                
                num_kps = kpts[:, 2].sum()
                if num_kps == 0:
                    hm[0, ct_int[1], ct_int[0]] = 0.9999
                    reg_mask[i] = 0
                
                for j in range(self.max_kpts):
                    kpt = kpts[j]
                    if kpt[2] > 0:
                        if self.split == 'train' and not self.opt.not_rand_crop:
                            kpt[:2] = affine_transform(kpt[:2], trans_output_rot)

                        if kpt[0] >= 0 and kpt[0] < output_res and \
                                kpt[1] >= 0 and kpt[1] < output_res:

                            hp_offset[i*self.max_kpts+j] = kpt[:2] - kpt[:2].astype(np.int32)

                            draw_gaussian(hm_hp[j], (int(kpt[0]), int(kpt[1])), radius)

                            kps[i][j*2:j*2+2] = kpt[0] - center[0], kpt[1] - center[1]

                            hp_mask[i*self.max_kpts+j] = 1

                            kps_mask[i][j*2:j*2+2] = 1

                            hp_ind[i*self.max_kpts + j] = int(kpt[1])*output_res + int(kpt[0])

                            if self.opt.dense_hp:
                                draw_dense_reg(dense_kps[j], hm[0], ct_int, kps[i][j*2:j*2+2], radius, True)
                                draw_gaussian(dense_kps_mask[j], ct_int, radius)
           
            draw_gaussian(hm[0], ct_int, sigma)
            gt_det.append([center[0] - w / 2, center[1] - h / 2,
                           center[0] + w / 2, center[1] + h / 2, 1] + kpts[:, :2].reshape(
                self.max_kpts * 2).tolist() + [0])
        
        ret = {'input': image, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,
               'hps': kps, 'hps_mask': kps_mask}
        if self.opt.dense_hp:
          dense_kps = dense_kps.reshape(self.max_kpts * 2, output_res, output_res)
          dense_kps_mask = dense_kps_mask.reshape(
            self.max_kpts, 1, output_res, output_res)
          dense_kps_mask = np.concatenate([dense_kps_mask, dense_kps_mask], axis=1)
          dense_kps_mask = dense_kps_mask.reshape(
            self.max_kpts * 2, output_res, output_res)
          ret.update({'dense_hps': dense_kps, 'dense_hps_mask': dense_kps_mask})
          del ret['hps'], ret['hps_mask']
        if self.opt.reg_offset:
          ret.update({'reg': reg})
        if self.opt.hm_hp:
          ret.update({'hm_hp': hm_hp})
        if self.opt.reg_hp_offset:
          ret.update({'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask})
        if self.opt.debug > 0 or not self.split == 'train':
          gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                   np.zeros((len(gt_det), len(gt_det[0])), dtype=np.float32)
          meta = {'gt_det': gt_det, 'img_id': image_id}
          ret['meta'] = meta
          
        """
        for key, value in ret.items():
            if value is None:
                print(f"this is returning a None: {key} for image id: {image_id}")  // for debugging
        """
        return ret
    

