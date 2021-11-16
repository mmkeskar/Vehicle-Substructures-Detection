#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 16:22:42 2021

@author: akshay
"""

import numpy as np
import math

def gaussian_heatmap(heatmap, center, sigma):
    """
    hm : np array
        the heatmap to be updated
    centers : np array
        the center of the detected keypoint (index format)
    sd : float
        standard deviation
    """
    h, w = heatmap.shape[0], heatmap.shape[1]
    radius = sigma*3*math.sqrt(2)
    ul = (max(0, int(center[0] - radius)), max(0, int(center[1] - radius)))
    br = (min(h-1, int(center[0] + radius)), min(w-1, int(center[1] + radius)))
    
    num_rows = br[0]+1-ul[0]
    num_cols = br[1]+1-ul[1]
    
    i = np.array(list(range(num_rows))*num_cols).reshape(num_cols, num_rows).transpose()
    j = np.array(list(range(num_cols))*num_rows).reshape(num_rows, num_cols)
    
    powers = ((i - center[0])**2 + (j - center[1])**2)/(2*(sigma**2))
    
    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
    
    mask = dist <= radius
    temp = mask*np.exp(-powers)
    
    heatmap[ul[0]:br[0]+1, ul[1]:br[1]+1] = np.maximum(heatmap[ul[0]:br[0]+1, ul[1]:br[1]+1], temp)
    

def get_radius(h, w, min_IOU=0.3):
    a = 1
    b = -(h+w)
    c = ((1-min_IOU)/(1+min_IOU))*h*w
    sqrt = math.sqrt(b**2 - 4*a*c)
    r = (-b + sqrt)/(2*a)
    return r
