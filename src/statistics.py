import matplotlib.pyplot as plt

from lib.models.model import create_model, load_model
import pycocotools.coco as coco
from lib.utils.image import gaussian_radius, draw_umich_gaussian
from lib.models.utils import _sigmoid
from lib.models.utils import _transpose_and_gather_feat
from lib.models.losses import FocalLoss
from lib.models.decode import _nms, _topk

import numpy as np
import torch
import cv2
import os

import pickle as pkl
import math

path_to_model = "../models/test_best_flip.pth"
heads = {'hm': 1, 'wh': 2, 'hps': 48, 'reg': 2, 'hm_hp': 24, 'hp_offset': 2}
model = create_model('dla_34', heads, 256)
model = load_model(model, path_to_model)
device = torch.device('cuda')
model = model.to(device)
model.eval()

anot_path = '../data/data-apollocar3d/annotations/apollo_keypoints_66_train.json'
my_coco = coco.COCO(anot_path)
my_imgs = my_coco.getImgIds()
keypoint_ids = [0, 1, 2, 3, 22, 23, 25, 26, 31, 32, 34, 35, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
max_objects = 49
max_kps = 24

all_cols = []
all_target_cols = []
input_res = 4096
for index in range(len(my_imgs)):
    print(index)
    image_id = my_imgs[index]
    file_path_ext = my_coco.loadImgs(image_id)[0]['file_name']
    img_pth_const = '../data/data-apollocar3d/train/images'
    image_path = os.path.join(img_pth_const, file_path_ext)
    inp = np.zeros((input_res, input_res, 3))
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    input_res0 = image.shape[0]
    input_res1 = image.shape[1]
    inp[:input_res0, :input_res1, :] = image
    inp = (inp.astype(np.float32)) / 255.
    inp = inp.transpose(2, 0, 1).reshape(1, 3, input_res, input_res)
    inp = torch.from_numpy(inp)
    inp = inp.to(device)
    torch.cuda.synchronize()

    ann_id = my_coco.getAnnIds([image_id])
    annotations = my_coco.loadAnns(ann_id)
    num_objs = min(max_objects, len(annotations))

    my_hm = np.zeros((1, 128, 128), dtype=np.float32)

    for i in range(num_objs):
        ann = annotations[i]
        bbox = ann['bbox']
        bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

        center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        ct_int = np.floor(center).astype(np.int)

        bbox = bbox / 4

        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        ct_int = np.floor(center).astype(np.int)

        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        sigma = radius / 3
        draw_umich_gaussian(my_hm[0], ct_int, sigma)

    my_hm = torch.from_numpy(my_hm).reshape(1, my_hm.shape[0], my_hm.shape[1], my_hm.shape[2])
    my_hm = _nms(my_hm)
    scores, inds, clses, ys, xs = _topk(my_hm, K=20)
    all_target_cols.append(xs.detach().cpu().numpy())

    with torch.no_grad():
        torch.cuda.synchronize()
        output = model(inp)[-1]
        output['hm'] = _sigmoid(output['hm'])
        output['hm'] = _nms(output['hm'])
        hm = output['hm']
        scores, inds, clses, ys, xs = _topk(hm, K=20)
        all_cols.append(xs.detach().cpu().numpy())

all_cols = np.array(all_cols).reshape(-1)
all_target_cols = np.array(all_target_cols).reshape(-1)
# print(all_cols)
file_name = 'hist_test_best_flip_0-4096_0-4096_all_cols_top_20'
file_name2 = 'hist_target_0-4096_0-4096_all_target_cols_top_20'
file_obj = open(file_name, 'wb')
file_obj2 = open(file_name2, 'wb')
pkl.dump(all_cols, file_obj)
pkl.dump(all_target_cols, file_obj2)

file_obj.close()
file_obj2.close()

plt.hist(all_cols)
plt.savefig("top20preds.png")
plt.show()

plt.hist(all_target_cols)
plt.savefig("top20targets.png")
plt.show()



