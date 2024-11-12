#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 15:32:47 2021
@author: akshay
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data

class Apollo3d(data.Dataset):
    num_classes = 1
    default_resolution = [512, 512]
    def __init__(self, opt, split):
        super(Apollo3d, self).__init__()
        
        self.data_dir = opt.data_dir
        self.opt = opt
        
        #TODO: change the file names and parsing info correctly
        if split == 'test':
            # self.annot_path = os.path.join(
                # self.data_dir, 'data-apollocar3d/annotations',
                # 'apollo_test2048_modified.json')
            # self.annot_path = os.path.join(
                # self.data_dir, 'data-apollocar3d/annotations',
                # 'apollo_testnosky_modified.json')
            # self.annot_path = os.path.join(
            #     self.data_dir, 'data-apollocar3d/annotations',
            #     'apollo_test_modified.json')
            if self.opt.task == 'offline_model1' or self.opt.task == 'cascaded':
                self.annot_path = os.path.join(
                    self.data_dir, 'data-apollocar3d/annotations',
                    'apollo_test_modified_offline_model1.json')
            else:
                self.annot_path = os.path.join(
                    self.data_dir, 'data-apollocar3d/annotations',
                    'apollo_test_modified.json')
        else:
          if opt.task == 'exdet':
            self.annot_path = os.path.join(
              self.data_dir, 'annotations', 
              'instances_extreme_{}2017.json').format(split)
          elif opt.task == 'vehint' or opt.task == 'vehint2' or opt.task == 'vehint_kptreg' or \
                  opt.task == 'vehint_bbpred':
            self.annot_path = os.path.join(
              self.data_dir, 'data-apollocar3d/annotations', 
              'apollo_{}_24_modkeypoints.json').format(split)
          elif opt.task == 'vehint_kptreg_2':
              self.annot_path = os.path.join(
                  self.data_dir, 'data-apollocar3d/annotations',
                  '24kptreg_{}.json').format(split)
          elif opt.task == 'offline_model1':
              self.annot_path = os.path.join(
                  self.data_dir, 'data-apollocar3d/annotations',
                  '6_offline_{}.json').format(split)
          elif opt.task == 'cascaded':
              self.annot_path = os.path.join(
                  self.data_dir, 'data-apollocar3d/annotations',
                  '6_offline_{}.json').format(split)
          else:
            self.annot_path = os.path.join(
              self.data_dir, 'annotations', 
              'apollo_{}_24_modkeypoints.json').format(split)
            
        self.image_path_const = os.path.join(self.data_dir, 'data-apollocar3d/train/images')
        if split == 'test':
            # self.image_path_const = os.path.join(self.data_dir, 'data-apollocar3d/test2048')
            # self.image_path_const = os.path.join(self.data_dir, 'data-apollocar3d/testnosky')
            self.image_path_const = os.path.join(self.data_dir, 'data-apollocar3d/test')
            if opt.single_image_test:
                self.image_path_const = os.path.join(self.data_dir, 'data-apollocar3d/test_single_picture')

        self.class_name = [
         "top_left_c_left_front_car_light",      # 0
        "bottom_left_c_left_front_car_light",   # 1
        "top_right_c_left_front_car_light",     # 2
        "bottom_right_c_left_front_car_light",  # 3

        "top_right_c_left_front_fog_light",     # 4
        "bottom_right_c_left_front_fog_light",  # 5
        "front_section_left_front_wheel",       # 6
        "center_left_front_wheel",              # 7
        "top_right_c_front_glass",              # 8
        "top_left_c_left_front_door",           # 9
        "bottom_left_c_left_front_door",        # 10
        "top_right_c_left_front_door",          # 11
        "middle_c_left_front_door",             # 12
        "front_c_car_handle_left_front_door",   # 13
        "rear_c_car_handle_left_front_door",    # 14
        "bottom_right_c_left_front_door",       # 15
        "top_right_c_left_rear_door",           # 16
        "front_c_car_handle_left_rear_door",    # 17
        "rear_c_car_handle_left_rear_door",     # 18
        "bottom_right_c_left_rear_door",        # 19
        "center_left_rear_wheel",               # 20
        "rear_section_left_rear_wheel",         # 21
        "top_left_c_left_rear_car_light",       # 22
        "bottom_left_c_left_rear_car_light",    # 23
        "top_left_c_rear_glass",                # 24
        "top_right_c_left_rear_car_light",      # 25
        "bottom_right_c_left_rear_car_light",   # 26
        "bottom_left_c_trunk",                  # 27
        "Left_c_rear_bumper",                   # 28
        "Right_c_rear_bumper",                  # 29
        "bottom_right_c_trunk",                 # 30
        "bottom_left_c_right_rear_car_light",   # 31
        "top_left_c_right_rear_car_light",      # 32
        "top_right_c_rear_glass",               # 33
        "bottom_right_c_right_rear_car_light",  # 34
        "top_right_c_right_rear_car_light",     # 35
        "rear_section_right_rear_wheel",        # 36
        "center_right_rear_wheel",              # 37
        "bottom_left_c_right_rear_car_door",    # 38
        "rear_c_car_handle_right_rear_car_door",    # 39
        "front_c_car_handle_right_rear_car_door",   # 40
        "top_left_c_right_rear_car_door",       # 41
        "bottom_left_c_right_front_car_door",   # 42
        "rear_c_car_handle_right_front_car_door",   # 43
        "front_c_car_handle_right_front_car_door",  # 44
        "middle_c_right_front_car_door",        # 45
        "top_left_c_right_front_car_door",      # 46
        "bottom_right_c_right_front_car_door",  # 47
        "top_right_c_right_front_car_door",     # 48
        "top_left_c_front_glass",               # 49
        "center_right_front_wheel",             # 50
        "front_section_right_front_wheel",      # 51
        "bottom_left_c_right_fog_light",        # 52
        "top_left_c_right_fog_light",           # 53
        "bottom_left_c_right_front_car_light",  # 54
        "top_left_c_right_front_car_light",     # 55
        "bottom_right_c_right_front_car_light",  # 56
        "top_right_c_right_front_car_light",     # 57
        "top_right_c_front_lplate",             # 58
        "top_left_c_front_lplate",              # 59
        "bottom_right_c_front_lplate",           # 60
        "bottom_left_c_front_lplate",          # 61
        "top_left_c_rear_lplate",               # 62
        "top_right_c_rear_lplate",              # 63
        "bottom_right_c_rear_lplate",           # 64
        "bottom_left_c_rear_lplate", ]            # 65
        
        self._valid_ids = [
         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
         13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
         24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 
         46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 
         57, 58, 59, 60, 61, 62, 63, 64, 65]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                         for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)

        self.split = split
        self.opt = opt
        self.min_area = 25

        print('==> initializing apollo3d {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        if opt.task == 'cascaded' or self.opt.kpts16_vehint:
            if opt.internal_breakdown_lf:
                self.annot_path_test = os.path.join(
                    self.data_dir, 'data-apollocar3d/annotations',
                    'left_headlight_cascaded_test_dataset.json')
            elif opt.internal_breakdown_lr:
                self.annot_path_test = os.path.join(
                    self.data_dir, 'data-apollocar3d/annotations',
                    'left_taillight_cascaded_test_dataset.json')
            elif opt.internal_breakdown_rr:
                self.annot_path_test = os.path.join(
                    self.data_dir, 'data-apollocar3d/annotations',
                    'right_taillight_cascaded_test_dataset.json')
            elif opt.internal_breakdown_rf:
                self.annot_path_test = os.path.join(
                    self.data_dir, 'data-apollocar3d/annotations',
                    'right_headlight_cascaded_test_dataset.json')
            else:
                self.annot_path_test = os.path.join(
                    self.data_dir, 'data-apollocar3d/annotations',
                    'apollo_cascade_model_final_test_annotation.json')
            self.coco_test = coco.COCO(self.annot_path_test)

        if opt.internal_breakdown_lf:
            if opt.task in ["vehint", "vehint_kptreg"]:
                self.annot_path_test = os.path.join(
                    self.data_dir, 'data-apollocar3d/annotations',
                    'left_headlight_vehint_test_dataset.json')
            elif opt.task == "offline_model1":
                self.annot_path_test = os.path.join(
                    self.data_dir, 'data-apollocar3d/annotations',
                    'left_headlight_offline1_test_dataset.json')
            self.coco_test = coco.COCO(self.annot_path_test)
        if opt.internal_breakdown_lr:
            if opt.task in ["vehint", "vehint_kptreg"]:
                self.annot_path_test = os.path.join(
                    self.data_dir, 'data-apollocar3d/annotations',
                    'left_taillight_vehint_test_dataset.json')
            elif opt.task == "offline_model1":
                self.annot_path_test = os.path.join(
                    self.data_dir, 'data-apollocar3d/annotations',
                    'left_taillight_offline1_test_dataset.json')
            self.coco_test = coco.COCO(self.annot_path_test)
        if opt.internal_breakdown_rr:
            if opt.task in ["vehint", "vehint_kptreg"]:
                self.annot_path_test = os.path.join(
                    self.data_dir, 'data-apollocar3d/annotations',
                    'right_taillight_vehint_test_dataset.json')
            elif opt.task == "offline_model1":
                self.annot_path_test = os.path.join(
                    self.data_dir, 'data-apollocar3d/annotations',
                    'right_taillight_offline1_test_dataset.json')
            self.coco_test = coco.COCO(self.annot_path_test)
        if opt.internal_breakdown_rf:
            if opt.task in ["vehint", "vehint_kptreg"]:
                self.annot_path_test = os.path.join(
                    self.data_dir, 'data-apollocar3d/annotations',
                    'right_headlight_vehint_test_dataset.json')
            elif opt.task == "offline_model1":
                self.annot_path_test = os.path.join(
                    self.data_dir, 'data-apollocar3d/annotations',
                    'right_headlight_offline1_test_dataset.json')
            self.coco_test = coco.COCO(self.annot_path_test)
        if opt.internal_breakdown_fl:
            if opt.task in ["vehint", "vehint_kptreg"]:
                self.annot_path_test = os.path.join(
                    self.data_dir, 'data-apollocar3d/annotations',
                    'front_license_vehint_test_dataset.json')
            elif opt.task == "offline_model1":
                self.annot_path_test = os.path.join(
                    self.data_dir, 'data-apollocar3d/annotations',
                    'front_license_offline1_test_dataset.json')
            self.coco_test = coco.COCO(self.annot_path_test)
        if opt.internal_breakdown_rl:
            if opt.task in ["vehint", "vehint_kptreg"]:
                self.annot_path_test = os.path.join(
                    self.data_dir, 'data-apollocar3d/annotations',
                    'rear_license_vehint_test_dataset.json')
            elif opt.task == "offline_model1":
                self.annot_path_test = os.path.join(
                    self.data_dir, 'data-apollocar3d/annotations',
                    'rear_license_offline1_test_dataset.json')
            self.coco_test = coco.COCO(self.annot_path_test)

        self.images = self.coco.getImgIds()
        
        self.num_samples = len(self.images)
        
        self.max_objects = 49  # found using python script
        self.num_kpts = 24
        self.num_cats = 1
        self.keypoint_ids = [0, 1, 2, 3, 22, 23, 25, 26, 31, 32, 34, 35, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
        self.max_kpts = len(self.keypoint_ids)

        if opt.task == 'vehint_kptreg' or opt.task == 'vehint_kptreg_2' \
                or opt.task == 'offline_model1' or opt.task == 'cascaded':
            self.num_kpts = 6
            self.num_int_kpts = 4
            self.max_internals = self.max_objects * self.num_kpts

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
      return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        if self.opt.task == 'vehint' or self.opt.task == 'vehint_kptreg':
            # import pdb; pdb.set_trace()
            detections = []
            for image_id in all_bboxes:
                for cls_ind in all_bboxes[image_id]:
                    category_id = 1
                    for dets in all_bboxes[image_id][cls_ind]:
                        bbox = dets[:4]
                        bbox[2] -= bbox[0]
                        bbox[3] -= bbox[1]
                        score = dets[4]
                        bbox_out = list(map(self._to_float, bbox))
                        if self.opt.kpts16_vehint:
                            keypoints = np.concatenate([
                                np.array(dets[5:37], dtype=np.float32).reshape(-1, 2),
                                np.ones((16, 1), dtype=np.float32)], axis=1).reshape(48).tolist()
                            num_keypoints = sum(keypoints[2:48:3])
                        elif self.opt.internal_breakdown_lf:
                            keypoints = np.concatenate([
                                np.array(dets[5:13], dtype=np.float32).reshape(-1, 2),
                                np.ones((4, 1), dtype=np.float32)], axis=1).reshape(12).tolist()
                            num_keypoints = sum(keypoints[2:12:3])
                        elif self.opt.internal_breakdown_lr:
                            keypoints = np.concatenate([
                                np.array(dets[13:21], dtype=np.float32).reshape(-1, 2),
                                np.ones((4, 1), dtype=np.float32)], axis=1).reshape(12).tolist()
                            num_keypoints = sum(keypoints[2:12:3])
                        elif self.opt.internal_breakdown_rr:
                            keypoints = np.concatenate([
                                np.array(dets[21:29], dtype=np.float32).reshape(-1, 2),
                                np.ones((4, 1), dtype=np.float32)], axis=1).reshape(12).tolist()
                            num_keypoints = sum(keypoints[2:12:3])
                        elif self.opt.internal_breakdown_rf:
                            keypoints = np.concatenate([
                                np.array(dets[29:37], dtype=np.float32).reshape(-1, 2),
                                np.ones((4, 1), dtype=np.float32)], axis=1).reshape(12).tolist()
                            num_keypoints = sum(keypoints[2:12:3])
                        elif self.opt.internal_breakdown_fl:
                            keypoints = np.concatenate([
                                np.array(dets[37:45], dtype=np.float32).reshape(-1, 2),
                                np.ones((4, 1), dtype=np.float32)], axis=1).reshape(12).tolist()
                            num_keypoints = sum(keypoints[2:12:3])
                        elif self.opt.internal_breakdown_rl:
                            keypoints = np.concatenate([
                                np.array(dets[45:53], dtype=np.float32).reshape(-1, 2),
                                np.ones((4, 1), dtype=np.float32)], axis=1).reshape(12).tolist()
                            num_keypoints = sum(keypoints[2:12:3])
                        else:
                            keypoints = np.concatenate([
                              np.array(dets[5:53], dtype=np.float32).reshape(-1, 2),
                              np.ones((24, 1), dtype=np.float32)], axis=1).reshape(72).tolist()
                            num_keypoints = sum(keypoints[2:72:3])
                        keypoints  = list(map(self._to_float, keypoints))

                        detection = {
                            "image_id": int(image_id),
                            "category_id": int(category_id),
                            "bbox": bbox_out,
                            "score": float("{:.2f}".format(score)),
                            "num_keypoints": num_keypoints,
                            "keypoints": keypoints
                        }
                        detections.append(detection)
                        print(f"detections shape: {len(detections[0]['keypoints'])}")
        elif self.opt.task == 'offline_model1':
            detections = []
            for image_id in all_bboxes:
                for cls_ind in all_bboxes[image_id]:
                    category_id = 1
                    for dets in all_bboxes[image_id][cls_ind]:
                        bbox = dets[:4]
                        bbox[2] -= bbox[0]
                        bbox[3] -= bbox[1]
                        score = dets[4]
                        bbox_out = list(map(self._to_float, bbox))
                        if self.opt.internal_breakdown_lf:
                            keypoints = np.concatenate([
                                np.array(dets[5:7], dtype=np.float32).reshape(-1, 2),
                                np.ones((1, 1), dtype=np.float32)], axis=1).reshape(3).tolist()
                            num_keypoints = sum(keypoints[2:12:3])
                        elif self.opt.internal_breakdown_lr:
                            keypoints = np.concatenate([
                                np.array(dets[7:9], dtype=np.float32).reshape(-1, 2),
                                np.ones((1, 1), dtype=np.float32)], axis=1).reshape(3).tolist()
                            num_keypoints = sum(keypoints[2:12:3])
                        elif self.opt.internal_breakdown_rr:
                            keypoints = np.concatenate([
                                np.array(dets[9:11], dtype=np.float32).reshape(-1, 2),
                                np.ones((1, 1), dtype=np.float32)], axis=1).reshape(3).tolist()
                            num_keypoints = sum(keypoints[2:12:3])
                        elif self.opt.internal_breakdown_rf:
                            keypoints = np.concatenate([
                                np.array(dets[11:13], dtype=np.float32).reshape(-1, 2),
                                np.ones((1, 1), dtype=np.float32)], axis=1).reshape(3).tolist()
                            num_keypoints = sum(keypoints[2:12:3])
                        elif self.opt.internal_breakdown_fl:
                            keypoints = np.concatenate([
                                np.array(dets[13:15], dtype=np.float32).reshape(-1, 2),
                                np.ones((1, 1), dtype=np.float32)], axis=1).reshape(3).tolist()
                            num_keypoints = sum(keypoints[2:12:3])
                        elif self.opt.internal_breakdown_rl:
                            keypoints = np.concatenate([
                                np.array(dets[15:17], dtype=np.float32).reshape(-1, 2),
                                np.ones((1, 1), dtype=np.float32)], axis=1).reshape(3).tolist()
                            num_keypoints = sum(keypoints[2:12:3])
                        else:
                            keypoints = np.concatenate([
                                np.array(dets[5:17], dtype=np.float32).reshape(-1, 2),
                                np.ones((6, 1), dtype=np.float32)], axis=1).reshape(18).tolist()
                            num_keypoints = sum(keypoints[2:18:3])
                        keypoints = list(map(self._to_float, keypoints))

                        detection = {
                            "image_id": int(image_id),
                            "category_id": int(category_id),
                            "bbox": bbox_out,
                            "score": float("{:.2f}".format(score)),
                            "num_keypoints": num_keypoints,
                            "keypoints": keypoints
                        }
                        detections.append(detection)
        elif self.opt.task == 'cascaded':
            detections = []
            for image_id in all_bboxes:
                for cls_ind in all_bboxes[image_id]:
                    category_id = 1
                    for dets in all_bboxes[image_id][cls_ind]:
                        bbox = dets[:4]
                        bbox[2] -= bbox[0]
                        bbox[3] -= bbox[1]
                        score = dets[4]
                        bbox_out = list(map(self._to_float, bbox))
                        if self.opt.internal_breakdown_lf:
                            keypoints = np.concatenate([
                                np.array(dets[5:13], dtype=np.float32).reshape(-1, 2),
                                np.ones((4, 1), dtype=np.float32)], axis=1).reshape(12).tolist()
                            num_keypoints = sum(keypoints[2:12:3])
                        elif self.opt.internal_breakdown_lr:
                            keypoints = np.concatenate([
                                np.array(dets[13:21], dtype=np.float32).reshape(-1, 2),
                                np.ones((4, 1), dtype=np.float32)], axis=1).reshape(12).tolist()
                            num_keypoints = sum(keypoints[2:12:3])
                        elif self.opt.internal_breakdown_rr:
                            keypoints = np.concatenate([
                                np.array(dets[21:29], dtype=np.float32).reshape(-1, 2),
                                np.ones((4, 1), dtype=np.float32)], axis=1).reshape(12).tolist()
                            num_keypoints = sum(keypoints[2:12:3])
                        elif self.opt.internal_breakdown_rf:
                            keypoints = np.concatenate([
                                np.array(dets[29:37], dtype=np.float32).reshape(-1, 2),
                                np.ones((4, 1), dtype=np.float32)], axis=1).reshape(12).tolist()
                            num_keypoints = sum(keypoints[2:12:3])
                        else:
                            keypoints = np.concatenate([
                                np.array(dets[5:37], dtype=np.float32).reshape(-1, 2),
                                np.ones((16, 1), dtype=np.float32)], axis=1).reshape(48).tolist()
                            num_keypoints = sum(keypoints[2:48:3])
                        keypoints = list(map(self._to_float, keypoints))

                        detection = {
                            "image_id": int(image_id),
                            "category_id": int(category_id),
                            "bbox": bbox_out,
                            "score": float("{:.2f}".format(score)),
                            "num_keypoints": num_keypoints,
                            "keypoints": keypoints
                        }
                        detections.append(detection)
        elif self.opt.task == 'vehint_kptreg_2':
            detections = []
            for image_id in all_bboxes:
                for cls_ind in all_bboxes[image_id]:
                    category_id = 1
                    for dets in all_bboxes[image_id][cls_ind]:
                        bbox = dets[:4]
                        bbox[2] -= bbox[0]
                        bbox[3] -= bbox[1]
                        score = dets[4]
                        bbox_out = list(map(self._to_float, bbox))
                        keypoints = np.concatenate([
                            np.array(dets[5:53], dtype=np.float32).reshape(-1, 2),
                            np.ones((24, 1), dtype=np.float32)], axis=1).reshape(72).tolist()
                        num_keypoints = sum(keypoints[2:72:3])
                        keypoints = list(map(self._to_float, keypoints))

                        detection = {
                            "image_id": int(image_id),
                            "category_id": int(category_id),
                            "bbox": bbox_out,
                            "score": float("{:.2f}".format(score)),
                            "num_keypoints": num_keypoints,
                            "keypoints": keypoints
                        }
                        detections.append(detection)
        return detections

    def __len__(self):
      return self.num_samples

    def save_results(self, results, save_dir):
      json.dump(self.convert_eval_format(results), 
                open('{}/results.json'.format(save_dir), 'w'))


    def run_eval(self, results, save_dir):
      # result_json = os.path.join(opt.save_dir, "results.json")
      # detections  = convert_eval_format(all_boxes)
      # json.dump(detections, open(result_json, "w"))
      self.save_results(results, save_dir)
      coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
      if self.opt.task == 'cascaded' or self.opt.kpts16_vehint:
          coco_eval = COCOeval(self.coco_test, coco_dets, "keypoints")
      else:
          coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
      # coco_eval = COCOeval(self.coco_test, coco_dets, "keypoints")
      coco_eval.evaluate()
      coco_eval.accumulate()
      coco_eval.summarize()

      """coco_eval = COCOeval(self.coco, coco_dets, "deviations")
      coco_eval.evaluate()"""


      if self.opt.task == 'cascaded' or self.opt.kpts16_vehint:
          coco_eval = COCOeval(self.coco_test, coco_dets, "bbox")
      else:
          coco_eval = COCOeval(self.coco, coco_dets, "bbox")
      coco_eval.evaluate()
      coco_eval.accumulate()
      coco_eval.summarize()

      """
      if self.opt.task == 'cascaded' or self.opt.kpts16_vehint:
          coco_eval = COCOeval(self.coco_test, coco_dets, "keypoints")
      else:
          coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
      coco_eval.evaluate()
      coco_eval.accumulate()
      coco_eval.summarize()
"""
