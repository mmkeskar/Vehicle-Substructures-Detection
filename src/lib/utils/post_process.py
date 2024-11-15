from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .image import transform_preds
from .ddd_utils import ddd2locrot


def get_pred_depth(depth):
  return depth

def get_alpha(rot):
  # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  # return rot[:, 0]
  idx = rot[:, 1] > rot[:, 5]
  alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
  alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + ( 0.5 * np.pi)
  return alpha1 * idx + alpha2 * (1 - idx)
  

def ddd_post_process_2d(dets, c, s, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  include_wh = dets.shape[2] > 16
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (opt.output_w, opt.output_h))
    classes = dets[i, :, -1]
    for j in range(opt.num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :3].astype(np.float32),
        get_alpha(dets[i, inds, 3:11])[:, np.newaxis].astype(np.float32),
        get_pred_depth(dets[i, inds, 11:12]).astype(np.float32),
        dets[i, inds, 12:15].astype(np.float32)], axis=1)
      if include_wh:
        top_preds[j + 1] = np.concatenate([
          top_preds[j + 1],
          transform_preds(
            dets[i, inds, 15:17], c[i], s[i], (opt.output_w, opt.output_h))
          .astype(np.float32)], axis=1)
    ret.append(top_preds)
  return ret

def ddd_post_process_3d(dets, calibs):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  for i in range(len(dets)):
    preds = {}
    for cls_ind in dets[i].keys():
      preds[cls_ind] = []
      for j in range(len(dets[i][cls_ind])):
        center = dets[i][cls_ind][j][:2]
        score = dets[i][cls_ind][j][2]
        alpha = dets[i][cls_ind][j][3]
        depth = dets[i][cls_ind][j][4]
        dimensions = dets[i][cls_ind][j][5:8]
        wh = dets[i][cls_ind][j][8:10]
        locations, rotation_y = ddd2locrot(
          center, alpha, dimensions, depth, calibs[0])
        bbox = [center[0] - wh[0] / 2, center[1] - wh[1] / 2,
                center[0] + wh[0] / 2, center[1] + wh[1] / 2]
        pred = [alpha] + bbox + dimensions.tolist() + \
               locations.tolist() + [rotation_y, score]
        preds[cls_ind].append(pred)
      preds[cls_ind] = np.array(preds[cls_ind], dtype=np.float32)
    ret.append(preds)
  return ret

def ddd_post_process(dets, c, s, calibs, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  dets = ddd_post_process_2d(dets, c, s, opt)
  dets = ddd_post_process_3d(dets, calibs)
  return dets


def ctdet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(
          dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret


def multi_pose_post_process(dets, c, s, h, w):
  # dets: batch x max_dets x 40
  # return list of 39 in image coord
  ret = []
  for i in range(dets.shape[0]):
    bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
    pts = transform_preds(dets[i, :, 5:39].reshape(-1, 2), c[i], s[i], (w, h))
    top_preds = np.concatenate(
      [bbox.reshape(-1, 4), dets[i, :, 4:5], 
       pts.reshape(-1, 34)], axis=1).astype(np.float32).tolist()
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret

def vehint_post_process(dets, input_res, kpts16=False):
  ret = []
  for i in range(dets.shape[0]):
    bbox = dets[i, :, :4]
    bbox[:, [0, 2]] = bbox[:, [0, 2]] * 4  # * (3384 / 512)
    bbox[:, [1, 3]] = bbox[:, [1, 3]] * 4   # * (2710 / 512)
    bbox = bbox.reshape(-1, 4)
    bbox = np.clip(bbox, 0, input_res)
    # print(f"bbox: {bbox}")
    if kpts16:
      pts = dets[i, :, 5:37].reshape(-1, 2)
    else:
      pts = dets[i, :, 5:53].reshape(-1, 2)
    pts[:, 0] = pts[:, 0] * 4
    pts[:, 1] = pts[:, 1] * 4
    if kpts16:
      pts = pts.reshape(-1, 32)
    else:
      pts = pts.reshape(-1, 48)
    pts = np.clip(pts, 0, input_res)
    if dets.shape[2] > 77:
      vis = dets[i, :, 53:77].reshape(-1, 24)
      top_preds = np.concatenate(
        [bbox, dets[i, :, 4:5], pts, vis], axis=1).astype(np.float32).tolist()
    else:
      top_preds = np.concatenate(
        [bbox, dets[i, :, 4:5], pts], axis=1).astype(np.float32).tolist()
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret

def offline_model1_post_process(dets, input_res):
  ret = []
  for i in range(dets.shape[0]):
    bbox = dets[i, :, :4]
    bbox[:, [0, 2]] = bbox[:, [0, 2]] * 4  # * (3384 / 512)
    bbox[:, [1, 3]] = bbox[:, [1, 3]] * 4   # * (2710 / 512)
    bbox = bbox.reshape(-1, 4)
    bbox = np.clip(bbox, 0, input_res)
    pts = dets[i, :, 5:17].reshape(-1, 2)
    pts[:, 0] = pts[:, 0] * 4
    pts[:, 1] = pts[:, 1] * 4
    pts = pts.reshape(-1, 12)
    pts = np.clip(pts, 0, input_res)
    if dets.shape[2] > 77:
      vis = dets[i, :, 53:77].reshape(-1, 24)
      top_preds = np.concatenate(
        [bbox, dets[i, :, 4:5], pts, vis], axis=1).astype(np.float32).tolist()
    else:
      top_preds = np.concatenate(
        [bbox, dets[i, :, 4:5], pts], axis=1).astype(np.float32).tolist()
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret

def cascaded_model_post_process(dets, input_res):
  ret = []
  for i in range(dets.shape[0]):
    bbox = dets[i, :, :4]
    bbox[:, [0, 2]] = bbox[:, [0, 2]] * 4  # * (3384 / 512)
    bbox[:, [1, 3]] = bbox[:, [1, 3]] * 4   # * (2710 / 512)
    bbox = bbox.reshape(-1, 4)
    bbox = np.clip(bbox, 0, input_res)
    pts = dets[i, :, 5:37]
    centers = dets[i, :, 37:49]

    lf = np.concatenate((pts[:, :2], pts[:, 4:6], pts[:, 2:4], pts[:, 6:8]), axis=1)
    lr = np.concatenate((pts[:, 8:10], pts[:, 12:14], pts[:, 10:12], pts[:, 14:16]), axis=1)
    rr = np.concatenate((pts[:, 20:22], pts[:, 16:18], pts[:, 22:24], pts[:, 18:20]), axis=1)
    rf = np.concatenate((pts[:, 28:30], pts[:, 24:26], pts[:, 30:32], pts[:, 26:28]), axis=1)
    pts = np.concatenate((lf, lr, rr, rf), axis=1)

    pts = np.clip(pts, 0, input_res)
    centers = np.clip(centers, 0, input_res)
    top_preds = np.concatenate(
      [bbox, dets[i, :, 4:5], pts, centers], axis=1).astype(np.float32).tolist()
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret

def vehint_tl_post_process(dets, input_res):
  ret = []
  for i in range(dets.shape[0]):
    bbox = dets[i, :, :4]
    bbox[:, [0, 2]] = bbox[:, [0, 2]] * 4  # * (3384 / 512)
    bbox[:, [1, 3]] = bbox[:, [1, 3]] * 4   # * (2710 / 512)
    bbox = bbox.reshape(-1, 4)
    bbox = np.clip(bbox, 0, input_res)
    # print(f"bbox: {bbox}")
    pts = dets[i, :, 5:21].reshape(-1, 2)
    pts[:, 0] = pts[:, 0] * 4
    pts[:, 1] = pts[:, 1] * 4
    pts = pts.reshape(-1, 16)
    pts = np.clip(pts, 0, input_res)
    if dets.shape[2] > 77:
      vis = dets[i, :, 53:77].reshape(-1, 24)
      top_preds = np.concatenate(
        [bbox, dets[i, :, 4:5], pts, vis], axis=1).astype(np.float32).tolist()
    else:
      top_preds = np.concatenate(
        [bbox, dets[i, :, 4:5], pts], axis=1).astype(np.float32).tolist()
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret

def vehint_kptreg_post_process(dets, input_res):
  ret = []
  for i in range(dets.shape[0]):
    bbox = dets[i, :, :4]
    bbox[:, [0, 2]] = bbox[:, [0, 2]] * 4  # * (3384 / 512)
    bbox[:, [1, 3]] = bbox[:, [1, 3]] * 4  # * (2710 / 512)
    bbox = bbox.reshape(-1, 4)
    bbox = np.clip(bbox, 0, input_res)
    # print(f"bbox: {bbox}")
    pts = dets[i, :, 5:17].reshape(-1, 2)
    pts[:, 0] = pts[:, 0] * 4
    pts[:, 1] = pts[:, 1] * 4
    pts = pts.reshape(-1, 12)
    pts = np.clip(pts, 0, input_res)

    kpts = dets[i, :, 5:53].reshape(-1, 2)
    kpts[:, 0] = kpts[:, 0] * 4
    kpts[:, 1] = kpts[:, 1] * 4
    kpts = kpts.reshape(-1, 48)
    kpts = np.clip(kpts, 0, input_res)

    if dets.shape[2] == 66:
      top_preds = np.concatenate(
        [bbox, dets[i, :, 4:5], pts, kpts], axis=1).astype(np.float32).tolist()
    elif dets.shape[2] > 77:
      vis = dets[i, :, 53:77].reshape(-1, 24)
      top_preds = np.concatenate(
        [bbox, dets[i, :, 4:5], pts, vis], axis=1).astype(np.float32).tolist()
    else:
      top_preds = np.concatenate(
        [bbox, dets[i, :, 4:5], pts], axis=1).astype(np.float32).tolist()
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret

def vehint_bbpred_post_process(dets, input_res):
  ret = []
  for i in range(dets.shape[0]):
    bbox = dets[i, :, :4]
    bbox[:, [0, 2]] = bbox[:, [0, 2]] * 4  # * (3384 / 512)
    bbox[:, [1, 3]] = bbox[:, [1, 3]] * 4  # * (2710 / 512)
    bbox = bbox.reshape(-1, 4)
    bbox = np.clip(bbox, 0, input_res)
    # print(f"bbox: {bbox}")
    pts = dets[i, :, 5:17].reshape(-1, 2)
    pts[:, 0] = pts[:, 0] * 4
    pts[:, 1] = pts[:, 1] * 4
    pts = pts.reshape(-1, 48)
    pts = np.clip(pts, 0, input_res)

    hp_bboxes = dets[i, :, 17:41]
    hp_bboxes[:, 0:24:2] = hp_bboxes[:, 0:24:2] * 4
    hp_bboxes[:, 1:24:2] = hp_bboxes[:, 1:24:2] * 4
    hp_bboxes = hp_bboxes.reshape(-1, 24)
    hp_bboxes = np.clip(hp_bboxes, 0, input_res)
    if dets.shape[2] > 77:
      vis = dets[i, :, 53:77].reshape(-1, 24)
      top_preds = np.concatenate(
        [bbox, dets[i, :, 4:5], pts, vis], axis=1).astype(np.float32).tolist()
    else:
      top_preds = np.concatenate(
        [bbox, dets[i, :, 4:5], pts, hp_bboxes], axis=1).astype(np.float32).tolist()
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret
