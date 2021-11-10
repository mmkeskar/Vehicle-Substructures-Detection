#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 15:20:51 2021

@author: akshay
"""

from vehint import VehIntDataset
import cv2
from torch.utils.data import DataLoader

data_dir = "/media/akshay/SSD1/data-apollocar3d/annotations/apollo_keypoints_66_train.json"
batch_size = 2

obj = VehIntDataset(data_dir)

sample = obj.__getitem__(0)

print(type(sample['labels']))


"""num_objects = []
for i in range(obj.__len__()):
    sample = obj.__getitem__(i)
    num_objects.append(sample['bboxes'].shape[0])
    """
#print(f"maximum objects is {max(num_objects)}")

def vizualize(sample):
    starting_points = []
    ending_points = []
    for car in sample['labels']:
        starting_points.append((int(car['bbox'][0]), int(car['bbox'][1])))
        ending_points.append((int(car['bbox'][0] + car['bbox'][2]), int(car['bbox'][1] + car['bbox'][3])))
    
    down_width = 1000
    down_height = 800
    down_points = (down_width, down_height)
        
    im = cv2.imread(sample['image_path'])
    
    im_copy = im.copy()
    for i in range(len(starting_points)):
        cv2.rectangle(im_copy, starting_points[i], ending_points[i], (0, 0, 255), thickness= 3, lineType=cv2.LINE_8)
        print(f"Starting point: {starting_points[i]}, Ending point: {ending_points[i]}")
        resize_down = cv2.resize(im_copy, down_points, interpolation= cv2.INTER_LINEAR)
        
    for car in sample['labels']:
        for keypoint in car['keypoints']:
            cv2.circle(im_copy, (int(keypoint[0]), int(keypoint[1])), 15, (255, 0, 0))
            resize_down = cv2.resize(im_copy, down_points, interpolation=cv2.INTER_LINEAR)
        
    cv2.imshow('imageRectangle', resize_down)
    cv2.waitKey(0)
    
def visualize_batch(batch):
    for j in range(batch['image'].shape[0]):
        print(f"batch size: {batch['image'].shape[0]}")
        bboxes = batch['bboxes'][j]
        starting_points = []
        ending_points = []
        for car in bboxes:
            if car[0] != -1:
                starting_points.append((car[0], car[1]))
                ending_points.append((car[0] + car[2], car[1] + car[3]))
    
        down_width = 1000
        down_height = 800
        down_points = (down_width, down_height)
            
        im = batch['image'][j].detach().numpy()
        
        im_copy = im.copy()
        for i in range(len(starting_points)):
            cv2.rectangle(im_copy, starting_points[i], ending_points[i], (0, 0, 255), thickness= 3, lineType=cv2.LINE_8)
            # print(f"Starting point: {starting_points[i]}, Ending point: {ending_points[i]}")
            resize_down = cv2.resize(im_copy, down_points, interpolation= cv2.INTER_LINEAR)
        
        cv2.imshow('imageRectangle' + str(j), resize_down)
        
    cv2.waitKey(0)
    
vizualize(sample)

dataloader = DataLoader(VehIntDataset(data_dir), batch_size=batch_size, shuffle=True, num_workers=0)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size())
    for label in sample_batched['labels']:
        print(label['bbox'].size())
        print(label['keypoints'].size())
    #visualize_batch(sample_batched)
    
