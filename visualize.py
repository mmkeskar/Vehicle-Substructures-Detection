#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:50:51 2021

@author: akshay
"""

import glob
import os
import cv2

#os.chdir("/media/akshay/SSD1/data-apollocar3d/train")
j = 0
for image in glob.glob("/media/akshay/SSD1/data-apollocar3d/train/images/*.jpg")[:1]:
    j += 1
    folders = image.split("/")
    image_name = folders[1].split(".")[0]
    key_file = "keypoints/" + image_name + "/*.txt"
    print(f"file name: {key_file}")
    starting_points = []
    ending_points = []
    for car in glob.glob(key_file):
        print(f"name of file: {car}")
        
        with open(car, "r") as file:
            lines = file.readlines()
            min_x = float(lines[0].split("\t")[1])
            print(f"minimum x: {min_x}")
            max_x = min_x
            max_y = float(lines[0].split("\t")[2].split("\n")[0])
            print(f"maximum y: {max_y}")
            min_y = max_y
            for line in lines:
                print("Entered for loop")
                if float(line.split("\t")[1]) < min_x:
                    min_x = float(line.split("\t")[1])
                if float(line.split("\t")[2].split("\n")[0]) > max_y:
                    max_y = float(line.split("\t")[2].split("\n")[0])
                if float(line.split("\t")[1]) > max_x:
                    max_x = float(line.split("\t")[1])
                if float(line.split("\t")[2].split("\n")[0]) < min_y:
                    min_y = float(line.split("\t")[2].split("\n")[0])
            print(f"final min_x: {min_x}, final min_y: {min_y}, final max_x: {max_x}, final max_y: {max_y}")
            padding_x = int(0.12*(max_x - min_x))
            padding_y = int(0.12*(max_y - min_y))
            starting_points.append((max(0, (int(min_x) - padding_x)), int(max_y) + padding_y))
            ending_points.append((int(max_x) + padding_x, max(0, int(min_y) - padding_y)))
    im = cv2.imread(image)
    print(f"type of object: {type(im)}")
    print(f"name of file of image: {image}")
    down_width = 1000
    down_height = 800
    down_points = (down_width, down_height)
    resize_down = cv2.resize(im, down_points, interpolation=cv2.INTER_LINEAR)

    cv2.imshow("previous" + str(j), resize_down)
    im_copy = im.copy()
    for i in range(len(starting_points)):
        cv2.rectangle(im_copy, starting_points[i], ending_points[i], (0, 0, 255), thickness= 3, lineType=cv2.LINE_8)
        print(f"Starting point: {starting_points[i]}, Ending point: {ending_points[i]}")
        resize_down = cv2.resize(im_copy, down_points, interpolation= cv2.INTER_LINEAR)
    
    cv2.imshow('imageRectangle' + str(j), resize_down)
    
cv2.waitKey(0)
      
    
            
    