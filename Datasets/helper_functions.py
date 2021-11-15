#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 16:22:42 2021

@author: akshay
"""

import numpy as np
import math

def gaussian_heatmap(hm, center, sd):
    """
    hm : np array
        the heatmap to be updated
    centers : np array
        the center of the detected keypoint (index format)
    sd : float
        standard deviation
    """
    for index, x in np.ndenumerate(hm):
        power = ((index[0] - center[0])**2 + (index[1] - center[1])**2)/(2*(sd**2))
        hm[index] = max(x, np.exp(-power))
        
    return hm

def get_radius(h, w, min_IOU=0.3):
    a = 1
    b = -(h+w)
    c = ((1-min_IOU)/(1+min_IOU))*h*w
    sqrt = math.sqrt(b**2 - 4*a*c)
    r = (-b + sqrt)/(2*a)
    return r