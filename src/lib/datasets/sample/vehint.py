# !/usr/bin/env python3
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

"""
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
"""
import math


class VehIntDataset(data.Dataset):
    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        image_id = self.images[index]
        file_path_ext = self.coco.loadImgs(image_id)[0]['file_name']
        image_path = os.path.join(self.image_path_const, file_path_ext)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        input_res = self.opt.input_res

        input_h = image.shape[0]
        input_w = image.shape[1]
        permissible_row = input_h - input_res  # 2710 - input_res
        permissible_col = input_w - input_res  # 3384 - input_res
        if self.opt.test:
            rand_row = 0
            rand_col = 0
            image = image[rand_row:rand_row + input_res, rand_col:rand_col + input_res, :]
            image_to_show = image.copy()
            image = (image.astype(np.float32) / 255.)
            image = image.transpose(2, 0, 1)
        else:
            if self.opt.add_lin_bias:
                rows = np.arange(0, permissible_row)
                probs = rows / np.sum(rows)
                rand_row = np.random.choice(rows, 1, p=probs)[0]
            elif self.opt.add_quad_bias:
                rows = np.arange(0, permissible_row) ** 2
                probs = rows / np.sum(rows)
                rand_row = np.random.choice(rows, 1, p=probs)[0]
            else:
                rand_row = np.random.randint(0, permissible_row)
            rand_col = np.random.randint(0, permissible_col)
            image = image[rand_row:rand_row + input_res, rand_col:rand_col + input_res, :]
            image_to_show = image.copy()
            image = (image.astype(np.float32) / 255.)
            image = image.transpose(2, 0, 1)
        output_res = self.opt.output_res

        ann_id = self.coco.getAnnIds([image_id])
        annotations = self.coco.loadAnns(ann_id)

        num_objs = min(self.max_objects, len(annotations))

        # a heatmap for the object centers
        hm = np.zeros((self.num_cats, output_res, output_res), dtype=np.float32)
        # a heatmap for all the keypoints
        hm_hp = np.zeros((self.num_kpts, output_res, output_res), dtype=np.float32)
        # the offsets of the keypoints from the object center points
        kps = np.zeros((self.max_objects, self.num_kpts * 2), dtype=np.float)
        reg_mask = np.zeros((self.max_objects), dtype=np.uint8)
        kps_mask = np.zeros((self.max_objects, self.num_kpts * 2), dtype=np.uint8)
        wh = np.zeros((self.max_objects, 2), dtype=np.float32)
        hp_mask = np.zeros((self.max_objects * self.num_kpts), dtype=np.int64)
        reg = np.zeros((self.max_objects, 2))
        hp_offset = np.zeros((self.max_objects * self.num_kpts, 2), dtype=np.float32)
        ind = np.zeros((self.max_objects), dtype=np.int64)
        hp_ind = np.zeros((self.max_objects * self.num_kpts), dtype=np.int64)
        kps_visibility = np.zeros((self.max_objects, self.num_kpts * 2), dtype=np.int64)
        kps_vis_mask = np.zeros((self.max_objects, self.num_kpts), dtype=np.int64)

        dense_kps = np.zeros((self.num_kpts, 2, output_res, output_res), dtype=np.float32)
        dense_kps_mask = np.zeros((self.num_kpts, output_res, output_res), dtype=np.float32)

        gt_det = []

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian

        starting_points = []
        ending_points = []
        centers = []

        for i in range(num_objs):
            ann = annotations[i]
            bbox = ann['bbox']
            bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            top_corner = rand_col <= bbox[0] < rand_col + input_res and rand_row <= bbox[1] < rand_row + input_res
            bottom_corner = rand_col <= bbox[2] < rand_col + input_res and rand_row <= bbox[3] < rand_row + input_res
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if (top_corner or bottom_corner) and w*h > self.min_area:
                if top_corner and bottom_corner:
                    # bbox is an array with bbox[0], bbox[1] giving top left and bbox[2], bbox[3] giving bottom right
                    row0, col0 = bbox[0] - rand_col, bbox[1] - rand_row
                    row1, col1 = bbox[2] - rand_col, bbox[3] - rand_row
                    bbox = np.array([row0, col0, row1, col1])
                else:
                    bbox[0] = max(min(rand_col + input_res - 1, bbox[0]), rand_col) - rand_col
                    bbox[1] = max(min(rand_row + input_res - 1, bbox[1]), rand_row) - rand_row
                    bbox[2] = max(min(rand_col + input_res - 1, bbox[2]), rand_col) - rand_col
                    bbox[3] = max(min(rand_row + input_res - 1, bbox[3]), rand_row) - rand_row

                starting_points.append((bbox[0], bbox[1]))
                ending_points.append((bbox[2], bbox[3]))

                # divide coordinated by down ratio to fit into output res dimensioned targets
                bbox = bbox / self.opt.down_ratio

                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

                center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                ct_int = np.floor(center).astype(np.int)

                ind[i] = ct_int[1] * output_res + ct_int[0]
                reg[i] = center - ct_int

                wh[i] = w, h

                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))

                reg_mask[i] = 1

                if len(ann['keypoints']) != 0:
                    kpts = np.array(ann['keypoints']).reshape(self.num_kpts, 3)
                    for kpt in kpts:
                        if rand_col <= kpt[0] < rand_col + input_res and rand_row <= kpt[1] < rand_row + input_res:
                            kpt[0], kpt[1] = kpt[0] - rand_col, kpt[1] - rand_row
                            # down sample
                            kpt[:2] = kpt[:2] / self.opt.down_ratio
                        else:
                            kpt[2] = 0

                    num_kps = kpts[:, 2].sum()
                    if num_kps == 0:
                        hm[0, ct_int[1], ct_int[0]] = 0.9999
                        reg_mask[i] = 0

                    for j in range(self.num_kpts):
                        kpt = kpts[j]
                        kps_visibility[i][j * 2] = 1 if kpt[2] == 2 else 0
                        kps_visibility[i][j * 2 + 1] = 0 if kpt[2] == 2 else 1
                        kps_vis_mask[i][j] = 1
                        if kpt[2] > 0:
                            centers.append((kpt[0], kpt[1]))
                            hp_offset[i * self.num_kpts + j] = kpt[:2] - kpt[:2].astype(np.int32)
                            kps[i][j * 2:j * 2 + 2] = kpt[0] - ct_int[0], kpt[1] - ct_int[1]
                            hp_mask[i * self.num_kpts + j] = 1
                            kps_mask[i][j * 2:j * 2 + 2] = 1
                            hp_ind[i * self.num_kpts + j] = int(kpt[1]) * output_res + int(kpt[0])
                            if self.opt.dense_hp:
                                draw_dense_reg(dense_kps[j], hm[0], ct_int, kps[i][j * 2:j * 2 + 2], radius, True)
                                draw_gaussian(dense_kps_mask[j], ct_int, radius)
                            draw_gaussian(hm_hp[j], (int(kpt[0]), int(kpt[1])), radius)
                else:
                    kpts = np.zeros((self.num_kpts, 3))

                draw_gaussian(hm[0], ct_int, radius)
                gt_det.append([center[0] - w / 2, center[1] - h / 2,
                               center[0] + w / 2, center[1] + h / 2, 1] + kpts[:, :2].reshape(
                    self.num_kpts * 2).tolist() + [0])

        ret = {'input': image, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,
               'hps': kps, 'hps_mask': kps_mask}
        if self.opt.dense_hp:
            dense_kps = dense_kps.reshape(self.num_kpts * 2, output_res, output_res)
            dense_kps_mask = dense_kps_mask.reshape(
                self.num_kpts, 1, output_res, output_res)
            dense_kps_mask = np.concatenate([dense_kps_mask, dense_kps_mask], axis=1)
            dense_kps_mask = dense_kps_mask.reshape(
                self.num_kpts * 2, output_res, output_res)
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
                np.zeros((1, 54), dtype=np.float32)
            meta = {'gt_det': gt_det, 'img_id': image_id}
            ret['meta'] = meta

        return ret

