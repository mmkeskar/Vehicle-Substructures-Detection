from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import torch
import torch.nn as nn
from .utils import _gather_feat, _transpose_and_gather_feat
from torchvision import models, transforms
import numpy as np
import cv2

from .helper_methods import make_cropped_image

def _nms(heat, kernel=3):  # TODO: play with this
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _left_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    shape = heat.shape 
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(1, heat.shape[0]):
        inds = (heat[i] >= heat[i - 1])
        ret[i] += ret[i - 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape) 

def _right_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    shape = heat.shape 
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(heat.shape[0] - 2, -1, -1):
        inds = (heat[i] >= heat[i +1])
        ret[i] += ret[i + 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape) 

def _top_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    heat = heat.transpose(3, 2) 
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(1, heat.shape[0]):
        inds = (heat[i] >= heat[i - 1])
        ret[i] += ret[i - 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape).transpose(3, 2)

def _bottom_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    heat = heat.transpose(3, 2) 
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(heat.shape[0] - 2, -1, -1):
        inds = (heat[i] >= heat[i + 1])
        ret[i] += ret[i + 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape).transpose(3, 2)

def _h_aggregate(heat, aggr_weight=0.1):
    return aggr_weight * _left_aggregate(heat) + \
           aggr_weight * _right_aggregate(heat) + heat

def _v_aggregate(heat, aggr_weight=0.1):
    return aggr_weight * _top_aggregate(heat) + \
           aggr_weight * _bottom_aggregate(heat) + heat

'''
# Slow for large number of categories
def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs
'''
def _topk_channel(scores, K=40):
      batch, cat, height, width = scores.size()
      
      topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

      topk_inds = topk_inds % (height * width)
      topk_ys   = (topk_inds / width).int().float()
      topk_xs   = (topk_inds % width).int().float()

      return topk_scores, topk_inds, topk_ys, topk_xs

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()  # row
    topk_xs   = (topk_inds % width).int().float()  # col
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def agnex_ct_decode(
    t_heat, l_heat, b_heat, r_heat, ct_heat, 
    t_regr=None, l_regr=None, b_regr=None, r_regr=None, 
    K=40, scores_thresh=0.1, center_thresh=0.1, aggr_weight=0.0, num_dets=1000
):
    batch, cat, height, width = t_heat.size()

    '''
    t_heat  = torch.sigmoid(t_heat)
    l_heat  = torch.sigmoid(l_heat)
    b_heat  = torch.sigmoid(b_heat)
    r_heat  = torch.sigmoid(r_heat)
    ct_heat = torch.sigmoid(ct_heat)
    '''
    if aggr_weight > 0: 
      t_heat = _h_aggregate(t_heat, aggr_weight=aggr_weight)
      l_heat = _v_aggregate(l_heat, aggr_weight=aggr_weight)
      b_heat = _h_aggregate(b_heat, aggr_weight=aggr_weight)
      r_heat = _v_aggregate(r_heat, aggr_weight=aggr_weight)
      
    # perform nms on heatmaps
    t_heat = _nms(t_heat)
    l_heat = _nms(l_heat)
    b_heat = _nms(b_heat)
    r_heat = _nms(r_heat)
      
      
    t_heat[t_heat > 1] = 1
    l_heat[l_heat > 1] = 1
    b_heat[b_heat > 1] = 1
    r_heat[r_heat > 1] = 1

    t_scores, t_inds, _, t_ys, t_xs = _topk(t_heat, K=K)
    l_scores, l_inds, _, l_ys, l_xs = _topk(l_heat, K=K)
    b_scores, b_inds, _, b_ys, b_xs = _topk(b_heat, K=K)
    r_scores, r_inds, _, r_ys, r_xs = _topk(r_heat, K=K)
      
    ct_heat_agn, ct_clses = torch.max(ct_heat, dim=1, keepdim=True)
      
    # import pdb; pdb.set_trace()

    t_ys = t_ys.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    t_xs = t_xs.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_ys = l_ys.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    l_xs = l_xs.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_ys = b_ys.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    b_xs = b_xs.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_ys = r_ys.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    r_xs = r_xs.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)

    box_ct_xs = ((l_xs + r_xs + 0.5) / 2).long()
    box_ct_ys = ((t_ys + b_ys + 0.5) / 2).long()

    ct_inds     = box_ct_ys * width + box_ct_xs
    ct_inds     = ct_inds.view(batch, -1)
    ct_heat_agn = ct_heat_agn.view(batch, -1, 1)
    ct_clses    = ct_clses.view(batch, -1, 1)
    ct_scores   = _gather_feat(ct_heat_agn, ct_inds)
    clses       = _gather_feat(ct_clses, ct_inds)

    t_scores = t_scores.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_scores = l_scores.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_scores = b_scores.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_scores = r_scores.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    ct_scores = ct_scores.view(batch, K, K, K, K)
    scores    = (t_scores + l_scores + b_scores + r_scores + 2 * ct_scores) / 6

    # reject boxes based on classes
    top_inds  = (t_ys > l_ys) + (t_ys > b_ys) + (t_ys > r_ys)
    top_inds = (top_inds > 0)
    left_inds  = (l_xs > t_xs) + (l_xs > b_xs) + (l_xs > r_xs)
    left_inds = (left_inds > 0)
    bottom_inds  = (b_ys < t_ys) + (b_ys < l_ys) + (b_ys < r_ys)
    bottom_inds = (bottom_inds > 0)
    right_inds  = (r_xs < t_xs) + (r_xs < l_xs) + (r_xs < b_xs)
    right_inds = (right_inds > 0)

    sc_inds = (t_scores < scores_thresh) + (l_scores < scores_thresh) + \
              (b_scores < scores_thresh) + (r_scores < scores_thresh) + \
              (ct_scores < center_thresh)
    sc_inds = (sc_inds > 0)

    scores = scores - sc_inds.float()
    scores = scores - top_inds.float()
    scores = scores - left_inds.float()
    scores = scores - bottom_inds.float()
    scores = scores - right_inds.float()

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    if t_regr is not None and l_regr is not None \
      and b_regr is not None and r_regr is not None:
        t_regr = _transpose_and_gather_feat(t_regr, t_inds)
        t_regr = t_regr.view(batch, K, 1, 1, 1, 2)
        l_regr = _transpose_and_gather_feat(l_regr, l_inds)
        l_regr = l_regr.view(batch, 1, K, 1, 1, 2)
        b_regr = _transpose_and_gather_feat(b_regr, b_inds)
        b_regr = b_regr.view(batch, 1, 1, K, 1, 2)
        r_regr = _transpose_and_gather_feat(r_regr, r_inds)
        r_regr = r_regr.view(batch, 1, 1, 1, K, 2)

        t_xs = t_xs + t_regr[..., 0]
        t_ys = t_ys + t_regr[..., 1]
        l_xs = l_xs + l_regr[..., 0]
        l_ys = l_ys + l_regr[..., 1]
        b_xs = b_xs + b_regr[..., 0]
        b_ys = b_ys + b_regr[..., 1]
        r_xs = r_xs + r_regr[..., 0]
        r_ys = r_ys + r_regr[..., 1]
    else:
        t_xs = t_xs + 0.5
        t_ys = t_ys + 0.5
        l_xs = l_xs + 0.5
        l_ys = l_ys + 0.5
        b_xs = b_xs + 0.5
        b_ys = b_ys + 0.5
        r_xs = r_xs + 0.5
        r_ys = r_ys + 0.5
      
    bboxes = torch.stack((l_xs, t_ys, r_xs, b_ys), dim=5)
    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)

    clses  = clses.contiguous().view(batch, -1, 1)
    clses  = _gather_feat(clses, inds).float()

    t_xs = t_xs.contiguous().view(batch, -1, 1)
    t_xs = _gather_feat(t_xs, inds).float()
    t_ys = t_ys.contiguous().view(batch, -1, 1)
    t_ys = _gather_feat(t_ys, inds).float()
    l_xs = l_xs.contiguous().view(batch, -1, 1)
    l_xs = _gather_feat(l_xs, inds).float()
    l_ys = l_ys.contiguous().view(batch, -1, 1)
    l_ys = _gather_feat(l_ys, inds).float()
    b_xs = b_xs.contiguous().view(batch, -1, 1)
    b_xs = _gather_feat(b_xs, inds).float()
    b_ys = b_ys.contiguous().view(batch, -1, 1)
    b_ys = _gather_feat(b_ys, inds).float()
    r_xs = r_xs.contiguous().view(batch, -1, 1)
    r_xs = _gather_feat(r_xs, inds).float()
    r_ys = r_ys.contiguous().view(batch, -1, 1)
    r_ys = _gather_feat(r_ys, inds).float()


    detections = torch.cat([bboxes, scores, t_xs, t_ys, l_xs, l_ys, 
                            b_xs, b_ys, r_xs, r_ys, clses], dim=2)

    return detections

def exct_decode(
    t_heat, l_heat, b_heat, r_heat, ct_heat, 
    t_regr=None, l_regr=None, b_regr=None, r_regr=None, 
    K=40, scores_thresh=0.1, center_thresh=0.1, aggr_weight=0.0, num_dets=1000
):
    batch, cat, height, width = t_heat.size()
    '''
    t_heat  = torch.sigmoid(t_heat)
    l_heat  = torch.sigmoid(l_heat)
    b_heat  = torch.sigmoid(b_heat)
    r_heat  = torch.sigmoid(r_heat)
    ct_heat = torch.sigmoid(ct_heat)
    '''

    if aggr_weight > 0:   
      t_heat = _h_aggregate(t_heat, aggr_weight=aggr_weight)
      l_heat = _v_aggregate(l_heat, aggr_weight=aggr_weight)
      b_heat = _h_aggregate(b_heat, aggr_weight=aggr_weight)
      r_heat = _v_aggregate(r_heat, aggr_weight=aggr_weight)
      
    # perform nms on heatmaps
    t_heat = _nms(t_heat)
    l_heat = _nms(l_heat)
    b_heat = _nms(b_heat)
    r_heat = _nms(r_heat)
      
    t_heat[t_heat > 1] = 1
    l_heat[l_heat > 1] = 1
    b_heat[b_heat > 1] = 1
    r_heat[r_heat > 1] = 1

    t_scores, t_inds, t_clses, t_ys, t_xs = _topk(t_heat, K=K)
    l_scores, l_inds, l_clses, l_ys, l_xs = _topk(l_heat, K=K)
    b_scores, b_inds, b_clses, b_ys, b_xs = _topk(b_heat, K=K)
    r_scores, r_inds, r_clses, r_ys, r_xs = _topk(r_heat, K=K)

    t_ys = t_ys.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    t_xs = t_xs.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_ys = l_ys.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    l_xs = l_xs.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_ys = b_ys.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    b_xs = b_xs.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_ys = r_ys.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    r_xs = r_xs.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)

    t_clses = t_clses.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_clses = l_clses.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_clses = b_clses.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_clses = r_clses.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    box_ct_xs = ((l_xs + r_xs + 0.5) / 2).long()
    box_ct_ys = ((t_ys + b_ys + 0.5) / 2).long()
    ct_inds = t_clses.long() * (height * width) + box_ct_ys * width + box_ct_xs
    ct_inds = ct_inds.view(batch, -1)
    ct_heat = ct_heat.view(batch, -1, 1)
    ct_scores = _gather_feat(ct_heat, ct_inds)

    t_scores = t_scores.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_scores = l_scores.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_scores = b_scores.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_scores = r_scores.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    ct_scores = ct_scores.view(batch, K, K, K, K)
    scores    = (t_scores + l_scores + b_scores + r_scores + 2 * ct_scores) / 6

    # reject boxes based on classes
    cls_inds = (t_clses != l_clses) + (t_clses != b_clses) + \
               (t_clses != r_clses)
    cls_inds = (cls_inds > 0)

    top_inds  = (t_ys > l_ys) + (t_ys > b_ys) + (t_ys > r_ys)
    top_inds = (top_inds > 0)
    left_inds  = (l_xs > t_xs) + (l_xs > b_xs) + (l_xs > r_xs)
    left_inds = (left_inds > 0)
    bottom_inds  = (b_ys < t_ys) + (b_ys < l_ys) + (b_ys < r_ys)
    bottom_inds = (bottom_inds > 0)
    right_inds  = (r_xs < t_xs) + (r_xs < l_xs) + (r_xs < b_xs)
    right_inds = (right_inds > 0)

    sc_inds = (t_scores < scores_thresh) + (l_scores < scores_thresh) + \
              (b_scores < scores_thresh) + (r_scores < scores_thresh) + \
              (ct_scores < center_thresh)
    sc_inds = (sc_inds > 0)

    scores = scores - sc_inds.float()
    scores = scores - cls_inds.float()
    scores = scores - top_inds.float()
    scores = scores - left_inds.float()
    scores = scores - bottom_inds.float()
    scores = scores - right_inds.float()

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    if t_regr is not None and l_regr is not None \
      and b_regr is not None and r_regr is not None:
        t_regr = _transpose_and_gather_feat(t_regr, t_inds)
        t_regr = t_regr.view(batch, K, 1, 1, 1, 2)
        l_regr = _transpose_and_gather_feat(l_regr, l_inds)
        l_regr = l_regr.view(batch, 1, K, 1, 1, 2)
        b_regr = _transpose_and_gather_feat(b_regr, b_inds)
        b_regr = b_regr.view(batch, 1, 1, K, 1, 2)
        r_regr = _transpose_and_gather_feat(r_regr, r_inds)
        r_regr = r_regr.view(batch, 1, 1, 1, K, 2)

        t_xs = t_xs + t_regr[..., 0]
        t_ys = t_ys + t_regr[..., 1]
        l_xs = l_xs + l_regr[..., 0]
        l_ys = l_ys + l_regr[..., 1]
        b_xs = b_xs + b_regr[..., 0]
        b_ys = b_ys + b_regr[..., 1]
        r_xs = r_xs + r_regr[..., 0]
        r_ys = r_ys + r_regr[..., 1]
    else:
        t_xs = t_xs + 0.5
        t_ys = t_ys + 0.5
        l_xs = l_xs + 0.5
        l_ys = l_ys + 0.5
        b_xs = b_xs + 0.5
        b_ys = b_ys + 0.5
        r_xs = r_xs + 0.5
        r_ys = r_ys + 0.5
      
    bboxes = torch.stack((l_xs, t_ys, r_xs, b_ys), dim=5)
    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)

    clses  = t_clses.contiguous().view(batch, -1, 1)
    clses  = _gather_feat(clses, inds).float()

    t_xs = t_xs.contiguous().view(batch, -1, 1)
    t_xs = _gather_feat(t_xs, inds).float()
    t_ys = t_ys.contiguous().view(batch, -1, 1)
    t_ys = _gather_feat(t_ys, inds).float()
    l_xs = l_xs.contiguous().view(batch, -1, 1)
    l_xs = _gather_feat(l_xs, inds).float()
    l_ys = l_ys.contiguous().view(batch, -1, 1)
    l_ys = _gather_feat(l_ys, inds).float()
    b_xs = b_xs.contiguous().view(batch, -1, 1)
    b_xs = _gather_feat(b_xs, inds).float()
    b_ys = b_ys.contiguous().view(batch, -1, 1)
    b_ys = _gather_feat(b_ys, inds).float()
    r_xs = r_xs.contiguous().view(batch, -1, 1)
    r_xs = _gather_feat(r_xs, inds).float()
    r_ys = r_ys.contiguous().view(batch, -1, 1)
    r_ys = _gather_feat(r_ys, inds).float()


    detections = torch.cat([bboxes, scores, t_xs, t_ys, l_xs, l_ys, 
                            b_xs, b_ys, r_xs, r_ys, clses], dim=2)


    return detections

def ddd_decode(heat, rot, depth, dim, wh=None, reg=None, K=40):
    batch, cat, height, width = heat.size()
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
      
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
      reg = _transpose_and_gather_feat(reg, inds)
      reg = reg.view(batch, K, 2)
      xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
      ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
      xs = xs.view(batch, K, 1) + 0.5
      ys = ys.view(batch, K, 1) + 0.5
      
    rot = _transpose_and_gather_feat(rot, inds)
    rot = rot.view(batch, K, 8)
    depth = _transpose_and_gather_feat(depth, inds)
    depth = depth.view(batch, K, 1)
    dim = _transpose_and_gather_feat(dim, inds)
    dim = dim.view(batch, K, 3)
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    xs = xs.view(batch, K, 1)
    ys = ys.view(batch, K, 1)
      
    if wh is not None:
        wh = _transpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, K, 2)
        detections = torch.cat(
            [xs, ys, scores, rot, depth, dim, wh, clses], dim=2)
    else:
        detections = torch.cat(
            [xs, ys, scores, rot, depth, dim, clses], dim=2)
      
    return detections

def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
      
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
      reg = _transpose_and_gather_feat(reg, inds)
      reg = reg.view(batch, K, 2)
      xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
      ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
      xs = xs.view(batch, K, 1) + 0.5
      ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
      wh = wh.view(batch, K, cat, 2)
      clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
      wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
      wh = wh.view(batch, K, 2)
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
      
    return detections

def multi_pose_decode(
    heat, wh, kps, reg=None, hm_hp=None, hp_offset=None, K=100):
  batch, cat, height, width = heat.size()
  num_internals = kps.shape[1] // 2
  # heat = torch.sigmoid(heat)
  # perform nms on heatmaps
  heat = _nms(heat)
  scores, inds, clses, ys, xs = _topk(heat, K=K)

  kps = _transpose_and_gather_feat(kps, inds)
  kps = kps.view(batch, K, num_internals * 2)
  kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_internals)
  kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_internals)
  if reg is not None:
    reg = _transpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
  else:
    xs = xs.view(batch, K, 1) + 0.5
    ys = ys.view(batch, K, 1) + 0.5
  wh = _transpose_and_gather_feat(wh, inds)
  wh = wh.view(batch, K, 2)
  clses  = clses.view(batch, K, 1).float()
  scores = scores.view(batch, K, 1)

  bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                      ys - wh[..., 1:2] / 2,
                      xs + wh[..., 0:1] / 2, 
                      ys + wh[..., 1:2] / 2], dim=2)
  if hm_hp is not None:
      hm_hp = _nms(hm_hp)
      thresh = 0.1
      kps = kps.view(batch, K, num_internals, 2).permute(
          0, 2, 1, 3).contiguous() # b x J x K x 2
      reg_kps = kps.unsqueeze(3).expand(batch, num_internals, K, K, 2)
      hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K) # b x J x K
      if hp_offset is not None:
          hp_offset = _transpose_and_gather_feat(
              hp_offset, hm_inds.view(batch, -1))
          hp_offset = hp_offset.view(batch, num_internals, K, 2)
          hm_xs = hm_xs + hp_offset[:, :, :, 0]
          hm_ys = hm_ys + hp_offset[:, :, :, 1]
      else:
          hm_xs = hm_xs + 0.5
          hm_ys = hm_ys + 0.5
        
      mask = (hm_score > thresh).float()
      hm_score = (1 - mask) * -1 + mask * hm_score
      hm_ys = (1 - mask) * (-10000) + mask * hm_ys
      hm_xs = (1 - mask) * (-10000) + mask * hm_xs
      hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
          2).expand(batch, num_internals, K, K, 2)
      dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
      min_dist, min_ind = dist.min(dim=3) # b x J x K
      hm_score = hm_score.gather(2, min_ind).unsqueeze(-1) # b x J x K x 1
      min_dist = min_dist.unsqueeze(-1)
      min_ind = min_ind.view(batch, num_internals, K, 1, 1).expand(
          batch, num_internals, K, 1, 2)
      hm_kps = hm_kps.gather(3, min_ind)
      hm_kps = hm_kps.view(batch, num_internals, K, 2)
      l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_internals, K, 1)
      t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_internals, K, 1)
      r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_internals, K, 1)
      b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_internals, K, 1)
      mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
             (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
             (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
      mask = (mask > 0).float().expand(batch, num_internals, K, 2)
      kps = (1 - mask) * hm_kps + mask * kps
      kps = kps.permute(0, 2, 1, 3).contiguous().view(
          batch, K, num_internals * 2)

      """
      int_ctrs_xs = torch.clamp(kps[..., ::2].view(batch, -1).int(), min=0, max=height - 1)  # shape: batch x K*6
      int_ctrs_ys = torch.clamp(kps[..., 1::2].view(batch, -1).int(), min=0, max=height - 1)
      reg_inds = (int_ctrs_ys.int() * height + int_ctrs_xs.int()).long()
      kps_vis = _transpose_and_gather_feat(kps_vis, reg_inds)
      visibility = kps_vis.min(2)[1].reshape(batch, K, num_internals, 1).expand(batch, K, num_internals, 2).\
          reshape(batch, K, num_internals * 2)

      kps = kps * visibility
      """

  detections = torch.cat([bboxes, scores, kps, clses], dim=2)
    
  return detections


def vehint_decode(images, hm, wh, kps, visibility_models=None, reg=None, hm_hp=None, hp_offset=None,
                  K=10, kpts16=False):
    batch, channels, height, width = hm.size()
    num_internals = kps.shape[1] // 2
    hm = _nms(hm)
    scores, inds, clses, ys, xs = _topk(hm, K=K)

    kps = _transpose_and_gather_feat(kps, inds)
    kps = kps.view(batch, K, num_internals * 2)
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_internals)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_internals)

    """
    kps_vis = _transpose_and_gather_feat(kps_vis, inds)
    kps_vis = kps_vis.view(batch, K, num_internals * 2)
    kps_vis = kps_vis.reshape(batch, -1, 2)
    kps_vis = F.softmax(kps_vis, dim=2)
    kps_vis = torch.min(kps_vis, 2)[1].reshape(batch, K, num_internals).float()
    """

    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    images = images.detach().cpu().numpy().transpose(0, 2, 3, 1)

    if visibility_models is not None:
        vis_images = np.zeros((bboxes.shape[0] * bboxes.shape[1], 3, 128, 128))
        num_images, num_objs = bboxes.shape[0], bboxes.shape[1]

        transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((128, 128)), transforms.ToTensor()])

        for i, image in enumerate(bboxes):
            for j, object in enumerate(image):
                box = object.detach().cpu().numpy()
                box = (np.array(box).astype(int)) * 4
                crop = images[i][box[1]: box[3], box[0]: box[2], :]
                crop = torch.tensor(crop)
                try:
                    crop = crop.permute(2, 0, 1)
                    crop = transform(crop)
                    vis_images[i * num_objs + j] = crop
                except:
                    continue

        vis_images = torch.from_numpy(vis_images)
        vis_images = vis_images.float()
        vis_preds = []
        for model in visibility_models:
            logits = model(vis_images)
            output = (logits > 0.5).float()
            for i in range(vis_images.shape[0]):
                if (vis_images[i] == 0).all():
                    output[i][0] = 0
            vis_preds.append(output)
        vis_preds = torch.stack(vis_preds)

        num_vis_models = len(visibility_models)
        vis_preds = torch.reshape(vis_preds, (num_vis_models, num_objs * num_images)).permute(1, 0)
        vis_preds = torch.reshape(vis_preds, (num_images, num_objs, num_vis_models))
        vis_pred_bbox = torch.reshape((vis_preds == 1).any(axis=2), (num_images, num_objs, 1)). \
            expand(num_images, num_objs, bboxes.shape[2])
        vis_pred_bbox = vis_pred_bbox.to(bboxes.device)
        bboxes = bboxes * vis_pred_bbox

        vis_pred_scores = torch.reshape((vis_preds == 1).any(axis=2), (num_images, num_objs, 1))
        vis_pred_scores = vis_pred_scores.to(scores.device)
        scores = scores * vis_pred_scores

    if hm_hp is not None:
        hm_hp = _nms(hm_hp)
        thresh = 0.1
        kps = kps.view(batch, K, num_internals, 2).permute(
            0, 2, 1, 3).contiguous()  # b x J x K x 2
        reg_kps = kps.unsqueeze(3).expand(batch, num_internals, K, K, 2)
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K)  # b x J x K

        if hp_offset is not None:
            hp_offset = _transpose_and_gather_feat(
                hp_offset, hm_inds.view(batch, -1))
            hp_offset = hp_offset.view(batch, num_internals, K, 2)
            hm_xs = hm_xs + hp_offset[:, :, :, 0]
            hm_ys = hm_ys + hp_offset[:, :, :, 1]
        else:
            hm_xs = hm_xs + 0.5
            hm_ys = hm_ys + 0.5

        if visibility_models is not None:
            vis_pred_kps = torch.reshape(vis_preds, (batch, K, num_vis_models, 1)).expand(batch, K, num_vis_models, 4)
            vis_pred_kps = torch.reshape(vis_pred_kps, (batch, K, num_internals))
            vis_pred_kps = vis_pred_kps.permute(0, 2, 1)
            vis_pred_kps = vis_pred_kps.to(hm_xs.device)
            hm_xs = hm_xs * vis_pred_kps
            hm_ys = hm_ys * vis_pred_kps

        mask = (hm_score > thresh).float()
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
            2).expand(batch, num_internals, K, K, 2)
        dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
        min_dist, min_ind = dist.min(dim=3)  # b x J x K
        hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_internals, K, 1, 1).expand(
            batch, num_internals, K, 1, 2)
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, num_internals, K, 2)
        l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_internals, K, 1)
        t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_internals, K, 1)
        r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_internals, K, 1)
        b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_internals, K, 1)
        mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
               (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
               (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
        mask = (mask > 0).float().expand(batch, num_internals, K, 2)
        kps = (1 - mask) * hm_kps + mask * kps
        kps = kps.permute(0, 2, 1, 3).contiguous().view(
            batch, K, num_internals * 2)
        if kpts16:
            kps = kps[:, :, :32]
        # kps = (kps * confidence).float()
    detections = torch.cat([bboxes, scores, kps, clses], dim=2)

    return detections


def offline_model1_decode(images, hm, wh, kps, visibility_models=None, reg=None, hm_hp=None,
                          hp_offset=None, K=2):
    batch, channels, height, width = hm.size()
    num_internals = kps.shape[1] // 2
    hm = _nms(hm)
    scores, inds, clses, ys, xs = _topk(hm, K=K)

    kps = _transpose_and_gather_feat(kps, inds)
    kps = kps.view(batch, K, num_internals * 2)
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_internals)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_internals)

    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    images = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    # images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

    if visibility_models and len(visibility_models) != 0:
        vis_images = np.zeros((bboxes.shape[0] * bboxes.shape[1], 3, 128, 128))
        num_images, num_objs = bboxes.shape[0], bboxes.shape[1]

        transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((128, 128)), transforms.ToTensor()])

        for i, image in enumerate(bboxes):
            for j, object in enumerate(image):
                box = object.detach().cpu().numpy()
                box = (np.array(box).astype(int)) * 4
                crop = images[i][box[1]: box[3], box[0]: box[2], :]
                crop = torch.tensor(crop)
                try:
                    crop = crop.permute(2, 0, 1)
                    crop = transform(crop)
                    vis_images[i * num_objs + j] = crop
                except Exception as e:
                    # print(f"error: {e}, {i}, {j}")
                    continue

        vis_images = torch.from_numpy(vis_images)
        vis_images = vis_images.float()
        vis_preds = []
        for model in visibility_models:
            logits = model(vis_images)
            output = (logits > 0.5).float()
            for i in range(vis_images.shape[0]):
                if (vis_images[i] == 0).all():
                    output[i][0] = 0
            vis_preds.append(output)
        vis_preds = torch.stack(vis_preds)

        num_vis_models = len(visibility_models)
        vis_preds = torch.reshape(vis_preds, (num_vis_models, num_objs * num_images)).permute(1, 0)
        vis_preds = torch.reshape(vis_preds, (num_images, num_objs, num_vis_models))
        vis_pred_bbox = torch.reshape((vis_preds == 1).any(axis=2), (num_images, num_objs, 1)). \
            expand(num_images, num_objs, bboxes.shape[2])
        vis_pred_bbox = vis_pred_bbox.to(bboxes.device)
        bboxes = bboxes * vis_pred_bbox

        vis_pred_scores = torch.reshape((vis_preds == 1).any(axis=2), (num_images, num_objs, 1))
        vis_pred_scores = vis_pred_scores.to(scores.device)
        scores = scores * vis_pred_scores

    if hm_hp is not None:
        hm_hp = _nms(hm_hp)
        thresh = 0.1
        kps = kps.view(batch, K, num_internals, 2).permute(
            0, 2, 1, 3).contiguous()  # b x J x K x 2
        reg_kps = kps.unsqueeze(3).expand(batch, num_internals, K, K, 2)
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K)  # b x J x K

        if hp_offset is not None:
            hp_offset = _transpose_and_gather_feat(
                hp_offset, hm_inds.view(batch, -1))
            hp_offset = hp_offset.view(batch, num_internals, K, 2)
            hm_xs = hm_xs + hp_offset[:, :, :, 0]
            hm_ys = hm_ys + hp_offset[:, :, :, 1]
        else:
            hm_xs = hm_xs + 0.5
            hm_ys = hm_ys + 0.5

        if visibility_models and len(visibility_models) != 0:
            vis_pred_kps = torch.reshape(vis_preds, (batch, K, num_vis_models, 1))
            vis_pred_kps = torch.reshape(vis_pred_kps, (batch, K, num_internals))
            vis_pred_kps = vis_pred_kps.permute(0, 2, 1)
            vis_pred_kps = vis_pred_kps.to(hm_xs.device)
            hm_xs = hm_xs * vis_pred_kps
            hm_ys = hm_ys * vis_pred_kps

        mask = (hm_score > thresh).float()
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
            2).expand(batch, num_internals, K, K, 2)
        dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
        min_dist, min_ind = dist.min(dim=3)  # b x J x K
        hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_internals, K, 1, 1).expand(
            batch, num_internals, K, 1, 2)
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, num_internals, K, 2)
        l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_internals, K, 1)
        t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_internals, K, 1)
        r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_internals, K, 1)
        b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_internals, K, 1)
        mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
               (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
               (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
        mask = (mask > 0).float().expand(batch, num_internals, K, 2)
        kps = (1 - mask) * hm_kps + mask * kps
        kps = kps.permute(0, 2, 1, 3).contiguous().view(
            batch, K, num_internals * 2)
        # kps = (kps * confidence).float()

    detections = torch.cat([bboxes, scores, kps, clses], dim=2)

    return detections


def offline_model2_decode(images, regression_models, dets):
    K = dets.shape[1]
    num_images = dets.shape[0]
    num_kpts = 6

    bboxes = torch.reshape(dets[:, :, :4], (num_images, K, 4))
    keypoints = torch.reshape(dets[:, :, 5:17], (num_images, K, 2 * num_kpts))

    vis_images_1, vis_images_2, vis_images_3, vis_images_4 = [], [], [], []
    tot_images = 0
    for i in range(num_images):
        for j in range(K):
            for k in range(0, 2 * num_kpts, 2):
                code = (k / 2) % 6
                padded_image = make_cropped_image(images[i].permute(1, 2, 0).detach().cpu().numpy(),
                                                  bboxes[i][j]*4, keypoints[i][j][k:k+2]*4)
                # padded_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)

                # Convert image to tensor
                padded_image = torch.tensor(padded_image)
                padded_image = padded_image.permute(2, 0, 1)
                # padded_image = torch.reshape(padded_image, (3, padded_image.size()[0], padded_image.size()[1]))
                padded_image = padded_image.float()
                tot_images += 1
                if code == 0:
                    vis_images_1.append(padded_image)
                elif code == 1:
                    vis_images_2.append(padded_image)
                elif code == 2:
                    vis_images_3.append(padded_image)
                elif code == 3:
                    vis_images_4.append(padded_image)
    vis_images_1 = torch.stack(vis_images_1)
    vis_images_2 = torch.stack(vis_images_2)
    vis_images_3 = torch.stack(vis_images_3)
    vis_images_4 = torch.stack(vis_images_4)
    vis_images = [vis_images_1, vis_images_2, vis_images_3, vis_images_4]

    outputs = []
    example = None
    for i, model in enumerate(regression_models):
        model_output = model(vis_images[i])
        outputs.append(64 * model_output)

    outputs = torch.stack(outputs)
    outputs = outputs.permute(1, 0, 2)
    keypoints = keypoints[:, :, :8]*4
    # deviations = torch.reshape(keypoints, (-1,))
    # deviations = deviations - deviations.type(torch.int64)
    # deviations = deviations.detach().cpu().numpy()
    # with open("deviations.txt", "ab") as f:
    #     np.savetxt(f, deviations)
    centers = keypoints.clone()
    keypoints = torch.reshape(keypoints, (num_images, K, len(regression_models), 1, 2))
    keypoints = keypoints.permute(0, 1, 2, 4, 3).expand(num_images, K, len(regression_models), 2, 4)
    keypoints = torch.reshape(keypoints.permute(0, 1, 2, 4, 3), (num_images, K, len(regression_models), 2*len(regression_models)))
    outputs = torch.reshape(outputs, (num_images, K, len(regression_models), 2*len(regression_models)))
    outputs = outputs.to(dets.device)
    outputs = outputs + keypoints
    outputs = torch.reshape(outputs, (num_images, K, len(regression_models)*2*len(regression_models)))

    dets = torch.cat([bboxes, dets[:, :, 4:5], outputs, centers, dets[:, :, 17:]], dim=2)
    return dets


def vehint_kptreg_decode(
        images, visibility_models, heat, wh, kps, kps_kps, reg,
        hm_hp, hp_offset, kps_kps_hm, hp_hp_offset, K=5):
    batch, cat, height, width = heat.size()
    num_internals = kps.shape[1] // 2  # kps shape: batch x 12 x 128 x 128
    num_kpts = kps_kps.shape[1] // 2   # kps_kps shape: batch x 8 x 128 x 128
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)  # xs shape: batch x K

    kps = _transpose_and_gather_feat(kps, inds)
    kps = kps.view(batch, K, num_internals * 2)
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_internals)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_internals)

    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)

    images = images.detach().cpu().numpy()[0].transpose(1, 2, 0)
    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

    vis_images = np.zeros((bboxes.shape[0]*bboxes.shape[1], 3, 128, 128))
    num_images, num_objs = bboxes.shape[0], bboxes.shape[1]

    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((128, 128)), transforms.ToTensor()])

    for i, image in enumerate(bboxes):
        for j, object in enumerate(image):
            box = object.detach().cpu().numpy()
            box = (np.array(box).astype(int)) * 4
            crop = images[box[1]: box[3], box[0]: box[2], :]
            crop = torch.tensor(crop)
            try:
                crop = crop.permute(2, 0, 1)
                crop = transform(crop)
                vis_images[i * num_objs + j] = crop
            except:
                continue

    vis_images = torch.from_numpy(vis_images)
    vis_images = vis_images.float()
    vis_preds = []
    for model in visibility_models:
        logits = model(vis_images)
        output = (logits > 0.5).float()
        for i in range(vis_images.shape[0]):
            if (vis_images[i] == 0).all():
                output[i][0] = 0
        vis_preds.append(output)
    vis_preds = torch.stack(vis_preds)

    num_vis_models = len(visibility_models)
    vis_preds = torch.reshape(vis_preds, (num_vis_models, num_objs * num_images)).permute(1, 0)
    vis_preds = torch.reshape(vis_preds, (num_images, num_objs, num_vis_models))
    vis_pred_bbox = torch.reshape((vis_preds == 1).any(axis=2), (num_images, num_objs, 1)). \
        expand(num_images, num_objs, bboxes.shape[2])
    vis_pred_bbox = vis_pred_bbox.to(bboxes.device)
    bboxes = bboxes * vis_pred_bbox

    if hm_hp is not None:
        hm_hp = _nms(hm_hp)
        thresh = 0.1
        kps = kps.view(batch, K, num_internals, 2).permute(
            0, 2, 1, 3).contiguous()  # b x J x K x 2
        reg_kps = kps.unsqueeze(3).expand(batch, num_internals, K, K, 2)
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K)  # b x J x K
        if hp_offset is not None:
            hp_offset = _transpose_and_gather_feat(
                hp_offset, hm_inds.view(batch, -1))
            hp_offset = hp_offset.view(batch, num_internals, K, 2)
            hm_xs = hm_xs + hp_offset[:, :, :, 0]
            hm_ys = hm_ys + hp_offset[:, :, :, 1]
        else:
            hm_xs = hm_xs + 0.5
            hm_ys = hm_ys + 0.5

        vis_pred_kps = vis_preds.permute(0, 2, 1)
        vis_pred_kps = vis_pred_kps.to(hm_xs.device)
        hm_xs = hm_xs * vis_pred_kps
        hm_ys = hm_ys * vis_pred_kps

        mask = (hm_score > thresh).float()
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
            2).expand(batch, num_internals, K, K, 2)
        dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
        min_dist, min_ind = dist.min(dim=3)  # b x J x K
        hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_internals, K, 1, 1).expand(
            batch, num_internals, K, 1, 2)
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, num_internals, K, 2)
        l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_internals, K, 1)
        t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_internals, K, 1)
        r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_internals, K, 1)
        b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_internals, K, 1)
        mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
               (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
               (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
        mask = (mask > 0).float().expand(batch, num_internals, K, 2)
        kps = (1 - mask) * hm_kps + mask * kps
        kps = kps.permute(0, 2, 1, 3).contiguous().view(
            batch, K, num_internals * 2)

    # co-ordinates of centers of internals
    int_ctrs_xs = torch.clamp(kps[..., ::2].view(batch, -1).int(), min=0, max=height-1)   # shape: batch x K*6
    int_ctrs_ys = torch.clamp(kps[..., 1::2].view(batch, -1).int(), min=0, max=height-1)
    reg_inds = (int_ctrs_ys.int() * height + int_ctrs_xs.int()).long()
    kps_kps = _transpose_and_gather_feat(kps_kps, reg_inds)
    kps_kps = kps_kps.view(batch, K, num_internals*num_kpts*2)
    kps_kps[..., ::2] += \
        int_ctrs_xs.view(batch, K*num_internals, 1).expand(batch, K*num_internals, num_kpts).reshape(batch, K, -1)
    kps_kps[..., 1::2] += \
        int_ctrs_ys.view(batch, K*num_internals, 1).expand(batch, K*num_internals, num_kpts).reshape(batch, K, -1)

    if hm_hp is not None:
        thresh = 0.1
        kps_kps_hm = _nms(kps_kps_hm)
        kps_kps = kps_kps.view(batch, K, num_kpts*num_internals, 2).permute(
            0, 2, 1, 3).contiguous()
        reg_kps_kps = kps_kps.unsqueeze(3).expand(batch, num_kpts*num_internals, K, K, 2)
        hm_hm_score, hm_hm_inds, hm_hm_ys, hm_hm_xs = _topk_channel(kps_kps_hm, K=K*num_internals)
        hm_hm_score = hm_hm_score.view(batch, num_kpts * num_internals, K)
        hm_hm_xs = hm_hm_xs.view(batch, num_kpts * num_internals, K)
        hm_hm_ys = hm_hm_ys.view(batch, num_kpts * num_internals, K)
        if hp_hp_offset is not None:
            hp_hp_offset = _transpose_and_gather_feat(hp_hp_offset, hm_hm_inds.view(batch, -1))
            hp_hp_offset = hp_hp_offset.view(batch, num_kpts * num_internals, K, 2)
            hm_hm_xs = hm_hm_xs + hp_hp_offset[:, :, :, 0]
            hm_hm_ys = hm_hm_ys + hp_hp_offset[:, :, :, 1]
        else:
            hm_hm_xs = hm_hm_xs + 0.5
            hm_hm_ys = hm_hm_ys + 0.5

        vis_pred_kps_kps = torch.reshape(vis_pred_kps, (batch, K, num_vis_models, 1)).expand(batch, K, num_vis_models, 4)
        vis_pred_kps_kps = torch.reshape(vis_pred_kps_kps, (batch, K, num_internals * num_kpts))
        vis_pred_kps_kps = vis_pred_kps_kps.permute(0, 2, 1)
        vis_pred_kps_kps = vis_pred_kps_kps.to(hm_xs.device)

        hm_hm_xs = hm_hm_xs * vis_pred_kps_kps
        hm_hm_ys = hm_hm_ys * vis_pred_kps_kps

        mask = (hm_hm_score > thresh).float()
        hm_hm_score = (1 - mask) * -1 + mask * hm_hm_score
        hm_hm_ys = (1 - mask) * (-10000) + mask * hm_hm_ys
        hm_hm_xs = (1 - mask) * (-10000) + mask * hm_hm_xs
        hm_hm_kps = torch.stack([hm_hm_xs, hm_hm_ys], dim=-1).unsqueeze(
            2).expand(batch, num_kpts*num_internals, K, K, 2)
        dist = (((reg_kps_kps - hm_hm_kps) ** 2).sum(dim=4) ** 0.5)
        min_dist, min_ind = dist.min(dim=3)  # b x J x K
        hm_hm_score = hm_hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_kpts*num_internals, K, 1, 1).expand(
            batch, num_kpts*num_internals, K, 1, 2)
        hm_hm_kps = hm_hm_kps.gather(3, min_ind)
        hm_hm_kps = hm_hm_kps.view(batch, num_kpts*num_internals, K, 2)
        l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_kpts*num_internals, K, 1)
        t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_kpts*num_internals, K, 1)
        r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_kpts*num_internals, K, 1)
        b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_kpts*num_internals, K, 1)
        mask = (hm_hm_kps[..., 0:1] < l) + (hm_hm_kps[..., 0:1] > r) + \
               (hm_hm_kps[..., 1:2] < t) + (hm_hm_kps[..., 1:2] > b) + \
               (hm_hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
        mask = (mask > 0).float().expand(batch, num_kpts*num_internals, K, 2)
        kps_kps = (1 - mask) * hm_hm_kps + mask * kps_kps
        kps_kps = kps_kps.permute(0, 2, 1, 3).contiguous().view(
            batch, K, num_kpts * num_internals * 2)
    detections = torch.cat([bboxes, scores, kps_kps, clses], dim=2)

    return detections

def vehint_bbpred_decode(
        heat, wh, kps, hp_bound_wh, reg=None, hp_bound_reg=None, hm_hp=None, hp_offset=None, K=100):
    batch, cat, height, width = heat.size()
    num_internals = kps.shape[1] // 2
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)

    kps = _transpose_and_gather_feat(kps, inds)
    kps = kps.view(batch, K, num_internals * 2)
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_internals)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_internals)

    # co-ordinates of centers of internals
    int_ctrs_xs = kps[..., ::2].view(batch, -1)  # shape: batch x K*6
    int_ctrs_ys = kps[..., 1::2].view(batch, -1)
    reg_inds = (int_ctrs_ys * height + int_ctrs_xs).long()

    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    if hm_hp is not None:
        hm_hp = _nms(hm_hp)
        thresh = 0.1
        kps = kps.view(batch, K, num_internals, 2).permute(
            0, 2, 1, 3).contiguous()  # b x J x K x 2
        reg_kps = kps.unsqueeze(3).expand(batch, num_internals, K, K, 2)
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K)  # b x J x K
        if hp_offset is not None:
            hp_offset = _transpose_and_gather_feat(
                hp_offset, hm_inds.view(batch, -1))
            hp_offset = hp_offset.view(batch, num_internals, K, 2)
            hm_xs = hm_xs + hp_offset[:, :, :, 0]
            hm_ys = hm_ys + hp_offset[:, :, :, 1]
        else:
            hm_xs = hm_xs + 0.5
            hm_ys = hm_ys + 0.5

        mask = (hm_score > thresh).float()
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
            2).expand(batch, num_internals, K, K, 2)
        dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
        min_dist, min_ind = dist.min(dim=3)  # b x J x K
        hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_internals, K, 1, 1).expand(
            batch, num_internals, K, 1, 2)
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, num_internals, K, 2)
        l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_internals, K, 1)
        t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_internals, K, 1)
        r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_internals, K, 1)
        b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_internals, K, 1)
        mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
               (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
               (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
        mask = (mask > 0).float().expand(batch, num_internals, K, 2)
        kps = (1 - mask) * hm_kps + mask * kps
        kps = kps.permute(0, 2, 1, 3).contiguous().view(
            batch, K, num_internals * 2)


        if reg is not None:
            hp_bound_reg = _transpose_and_gather_feat(hp_bound_reg, reg_inds)
            hp_bound_reg = hp_bound_reg.view(batch, K, -1)
            int_ctrs_xs = int_ctrs_xs.view(batch, K, -1) + reg[:, :, 0:num_internals:2]
            int_ctrs_ys = int_ctrs_ys.view(batch, K, -1) + reg[:, :, 1:num_internals:2]
        else:
            int_ctrs_xs = int_ctrs_xs.view(batch, K, -1) + 0.5
            int_ctrs_ys = int_ctrs_ys.view(batch, K, -1) + 0.5
        hp_bound_wh = _transpose_and_gather_feat(hp_bound_wh, reg_inds)
        hp_bound_wh = hp_bound_wh.view(batch, K, -1)

        # int_ctrs_xs shape: batch x K x 6
        # hp_bboxes shape : batch x K x 6*4
        hp_bboxes = torch.cat([int_ctrs_xs - hp_bound_wh[..., 0:num_internals:2] / 2,
                            int_ctrs_ys - hp_bound_wh[..., 1:num_internals:2] / 2,
                            int_ctrs_xs + hp_bound_wh[..., 0:num_internals:2] / 2,
                            int_ctrs_ys + hp_bound_wh[..., 1:num_internals:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, kps, hp_bboxes, clses], dim=2)
    # TODO: find out what metrics to use for evaluation and comparison with other models and depending on that
    # TODO: find score for the vehicle internals bounding boxes

    return detections
