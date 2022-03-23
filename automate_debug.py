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
import math

import matplotlib.pyplot as plt
import pickle as pkl

def exp_generator(image_to_show, input):
    input_h = image_to_show.shape[0]
    input_w = image_to_show.shape[1]
    pixel_r = input_h // 2
    pixel_c = input_w // 2
    yield image_to_show, input

    exp_img, exp_inp = image_to_show.copy(), input.copy()
    exp_img[:pixel_r, :, :] = 0
    exp_inp[:pixel_r, :, :] = 0
    yield exp_img, exp_inp

    exp_img, exp_inp = image_to_show.copy(), input.copy()
    exp_img[pixel_r:, :, :] = 0
    exp_inp[pixel_r:, :, :] = 0
    yield exp_img, exp_inp

    exp_img, exp_inp = image_to_show.copy(), input.copy()
    exp_img[:, pixel_c:, :] = 0
    exp_inp[:, pixel_c:, :] = 0
    yield exp_img, exp_inp

    exp_img, exp_inp = image_to_show.copy(), input.copy()
    exp_img[:, :pixel_c, :] = 0
    exp_inp[:, :pixel_c, :] = 0
    yield exp_img, exp_inp

    """
    exp_img, exp_inp = image_to_show.copy(), input.copy()
    exp_img[:100, :, :] = 0
    exp_img[412:, :, :] = 0
    exp_img[:, :100, :] = 0
    exp_img[:, 412:, :] = 0
    exp_inp[:100, :, :] = 0
    exp_inp[412:, :, :] = 0
    exp_inp[:, :100, :] = 0
    exp_inp[:, 412:, :] = 0
    yield exp_img, exp_inp

    exp_img, exp_inp = image_to_show, input
    exp_img[200:312, 200:312, :] = 0
    exp_inp[200:312, 200:312, :] = 0
    yield exp_img, exp_inp
    """

path_to_model = "../models/overfit_single_image.pth"
heads = {'hm': 1, 'wh': 2, 'hps': 34, 'reg': 2, 'hm_hp': 17, 'hp_offset': 2}
model = create_model('dla_34', heads, 256)
model = load_model(model, path_to_model)
device = torch.device('cuda')
model = model.to(device)
model.eval()

anot_path = '../data/coco/annotations/person_keypoints_train2017.json'                        #'../data/data-apollocar3d/annotations/apollo_keypoints_66_train.json'
my_coco = coco.COCO(anot_path)
my_imgs = my_coco.getImgIds()
max_objects = 32
max_kps = 17

input_res = 512
rand_row = 0
rand_col = 0

down_width = 512
down_height = 512
down_points = (down_width, down_height)

indices = range(len(my_imgs))
num_imgs = 10
probs = np.ones(len(my_imgs))*(1/len(my_imgs))
image_indices = [49870]             # np.random.choice(indices, num_imgs, p=probs)
for index in image_indices:
    image_id = my_imgs[index]
    file_path_ext = my_coco.loadImgs(image_id)[0]['file_name']
    img_pth_const = '../data/coco/train2017'             # '../data/data-apollocar3d/train/images'
    image_path = os.path.join(img_pth_const, file_path_ext)
    file_path_ext = "overfit"

    os.mkdir(file_path_ext)
    with open(file_path_ext + '/index.txt', 'w') as f:
        f.write(f"The index provided is: {index}")
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image.ndim != 3:
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    input_h = image.shape[0]
    input_w = image.shape[1]
    if input_h > input_res:
        end_row = input_res
    else:
        end_row = input_h
    if input_w > input_res:
        end_col = input_res
    else:
        end_col = input_w

    image_to_show = image[rand_row:end_row, rand_col:end_col, :]
    inp = np.zeros((input_res, input_res, 3))
    inp[:end_row - rand_row, :end_col - rand_col, :] = image_to_show

    inp = (inp.astype(np.float32)) / 255.

    centers = []
    ann_id = my_coco.getAnnIds([image_id])
    annotations = my_coco.loadAnns(ann_id)
    num_objs = min(max_objects, len(annotations))

    my_hm = np.zeros((1, 128, 128), dtype=np.float32)

    for i in range(num_objs):
        ann = annotations[i]
        bbox = ann['bbox']
        bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
        top_corner = rand_col <= bbox[0] < end_col and rand_row <= bbox[1] < end_row
        bottom_corner = rand_col <= bbox[2] < end_col and rand_row <= bbox[3] < end_row
        # veh_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        # center_cond = rand_col+8 <= veh_center[0] < input_res-8+rand_col and rand_row+8 <= veh_center[1] < input_res+rand_row-8
        if top_corner or bottom_corner:
            if top_corner and bottom_corner:
                # bbox is an array with bbox[0], bbox[1] giving top left and bbox[2], bbox[3] giving bottom right
                row0, col0 = bbox[0] - rand_col, bbox[1] - rand_row
                row1, col1 = bbox[2] - rand_col, bbox[3] - rand_row
                bbox = np.array([row0, col0, row1, col1])
            else:
                bbox[0] = max(min(end_col - 1, bbox[0]), rand_col) - rand_col
                bbox[1] = max(min(end_row - 1, bbox[1]), rand_row) - rand_row
                bbox[2] = max(min(end_col - 1, bbox[2]), rand_col) - rand_col
                bbox[3] = max(min(end_row - 1, bbox[3]), rand_row) - rand_row
            # print(bbox)

            # divide coordinated by down ratio to fit into output res dimensioned targets
            bbox = bbox / 4

            # print(bbox)

            w, h = abs(bbox[2] - bbox[0]), bbox[3] - bbox[1]

            center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            ct_int = np.floor(center).astype(np.int)

            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            sigma = radius / 3
            draw_umich_gaussian(my_hm[0], ct_int, sigma)
        """ 
        ann = annotations[i]
        bbox = ann['bbox']
        bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
        top_corner = rand_col <= bbox[0] < rand_col + input_res and rand_row <= bbox[1] < rand_row + input_res
        bottom_corner = rand_col <= bbox[2] < rand_col + input_res and rand_row <= bbox[3] < rand_row + input_res
        if top_corner or bottom_corner:
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
            # print(bbox)

            center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            ct_int = np.floor(center).astype(np.int)
            centers.append(ct_int)

            bbox = bbox / 4

            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

            center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            ct_int = np.floor(center).astype(np.int)

            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            sigma = radius / 3
            draw_umich_gaussian(my_hm[0], ct_int, sigma)
            """

    for center in centers:
        print(center)
        cv2.circle(image_to_show, center, 3, (255, 0, 0), 2)
        resize_down = cv2.resize(image_to_show, down_points, interpolation=cv2.INTER_LINEAR)

    fig = plt.figure()
    plt.imshow(my_hm[0])
    file_name = file_path_ext + "/target_heatmap.png"
    plt.savefig(file_name)

    fig = plt.figure()
    plt.imshow(image_to_show)
    file_name = file_path_ext + "/targets.png"
    plt.savefig(file_name)

    gen = exp_generator(image_to_show, inp)
    counter = 0

    while True:
        try:
            counter += 1
            image, inp = next(gen)
            inp = inp.transpose(2, 0, 1).reshape(1, 3, input_res, input_res)
            inp = torch.from_numpy(inp)
            inp = inp.to(device)
            torch.cuda.synchronize()
            with torch.no_grad():
                torch.cuda.synchronize()
                output = model(inp)[-1]
                print(output['hm'][0][0].shape)
                output['hm'] = _sigmoid(output['hm'])
                fig = plt.figure()
                plt.imshow(output['hm'][0][0].detach().cpu().numpy())
                file_name = file_path_ext + "/predicted_sig_heat_" + str(counter) + ".png"
                plt.savefig(file_name)
                output['hm'] = _nms(output['hm'])
                hm = output['hm']
                scores, inds, clses, ys, xs = _topk(hm, K=10)
                for i in range(len(ys[0])):
                    cv2.circle(image, (int(xs[0][i] * 4), int(ys[0][i] * 4)), 3, (0, 255, 0), 2)
                    resize_down = cv2.resize(image, down_points, interpolation=cv2.INTER_LINEAR)
                fig = plt.figure()
                plt.imshow(image)
                file_name = file_path_ext + "/predictions_" + str(counter) + ".png"
                plt.savefig(file_name)

                fig = plt.figure()
                plt.imshow(hm[0][0].detach().cpu().numpy())
                file_name = file_path_ext + "/predicted_heatmap_" + str(counter) + ".png"
                plt.savefig(file_name)
        except StopIteration:
            break



