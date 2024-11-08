from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import torch.nn.functional as F
from torchvision import models, transforms

from models.decode import multi_pose_decode
from models.model import create_model, load_model
from utils.image import get_affine_transform
from utils.post_process import vehint_tl_post_process
from utils.debugger import Debugger
from models.decode import _nms, _topk, _topk_channel
from models.utils import _gather_feat, _transpose_and_gather_feat

import matplotlib.pyplot as plt

class VehintTaillightDetector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')

        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.max_per_image = 100
        self.num_classes = opt.num_classes
        self.scales = opt.test_scales
        self.opt = opt
        self.pause = True

        self.edges = [[0, 1], [1, 3], [2, 3], [0, 2],
                      [4, 5], [4, 6], [5, 7], [6, 7]]

        self.colors = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255),
                       (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0)]

    def pre_process(self, image):
        input_res = 512  # self.opt.input_res
        """
        image = cv2.resize(image, (input_res, input_res), interpolation=cv2.INTER_AREA)
        image = (image.astype(np.float32) / 255.)
        inp_height, inp_width = self.opt.input_h, self.opt.input_w
        images = image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)"""

        input_h = image.shape[0]
        input_w = image.shape[1]

        permissible_row = input_h - input_res  # 2710 - input_res
        permissible_col = input_w - input_res  # 3384 - input_res
        rand_row = 0 # np.random.randint(0, permissible_row) # 1680 #
        rand_col = 0 # np.random.randint(0, permissible_col) # 640 #

        if input_h > input_res:
            end_row = input_res + rand_row
        else:
            end_row = input_h
        if input_w > input_res:
            end_col = input_res + rand_col
        else:
            end_col = input_w

        image = image[rand_row:end_row, rand_col:end_col, :]

        image = (image.astype(np.float32) / 255.)
        images = image.transpose(2, 0, 1).reshape(1, 3, image.shape[0], image.shape[1])
        images = torch.from_numpy(images)
        return images

    def merge_outputs(self, detections):
        results = {}
        results[1] = np.concatenate(
            [detection[1] for detection in detections], axis=0).astype(np.float32)
        if self.opt.nms or len(self.opt.test_scales) > 1:
            soft_nms_39(results[1], Nt=0.5, method=2)
        results[1] = results[1].tolist()
        return results

    def vehint_decode(self, images, visibility_models, hm, wh, kps, reg=None, hm_hp=None, hp_offset=None, K=2):
        print(f"value of K: {K}")
        batch, channels, height, width = hm.size()
        num_internals = kps.shape[1]//2
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
        images = images.detach().cpu().numpy()[0].transpose(1, 2, 0)

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
                    print(f"shape of crop: {crop.shape}, {i}, {j}")
                    crop = crop.permute(2, 0, 1)
                    crop = transform(crop)
                    vis_images[i*num_objs + j] = crop
                except Exception as e:
                    print(f"error: {e}, {i}, {j}")
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
        vis_pred_bbox = torch.reshape((vis_preds == 1).any(axis=2), (num_images, num_objs, 1)).\
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
            reg_kps = reg_kps[:, 4:12, :, :, :]
            hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K) # b x J x K
            hm_score = hm_score[:, 4:12, :]
            hm_xs, hm_ys = hm_xs[:, 4:12, :], hm_ys[:, 4:12, :]

            if hp_offset is not None:
                hp_offset = _transpose_and_gather_feat(
                    hp_offset, hm_inds.view(batch, -1))
                hp_offset = hp_offset.view(batch, num_internals, K, 2)
                hp_offset = hp_offset[:, 4:12, :, :]
                hm_xs = hm_xs + hp_offset[:, :, :, 0]
                hm_ys = hm_ys + hp_offset[:, :, :, 1]
            else:
                hm_xs = hm_xs + 0.5
                hm_ys = hm_ys + 0.5

            num_internals = 8
            kps = kps[:, 4:12, :, :]

            vis_pred_kps = torch.reshape(vis_preds, (batch, K, num_vis_models, 1)).expand(batch, K, num_vis_models, 4)
            vis_pred_kps = torch.reshape(vis_pred_kps, (batch, K, num_internals))
            vis_pred_kps = vis_pred_kps.permute(0, 2, 1)
            vis_pred_kps = vis_pred_kps.to(hm_xs.device)
            print(f"shape of hm_xs: {hm_xs.shape}")
            print(f"shape of vis_pred_kps: {vis_pred_kps.shape}")
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
            print(f"shape of kps: {kps.shape}")
            print(f"mask shape: {mask.shape}")
            print(f"hm_kps shape: {hm_kps.shape}")
            mask = (mask > 0).float().expand(batch, num_internals, K, 2)
            kps = (1 - mask) * hm_kps + mask * kps
            kps = kps.permute(0, 2, 1, 3).contiguous().view(
                batch, K, num_internals * 2)
            # kps = (kps * confidence).float()
        detections = torch.cat([bboxes, scores, kps, clses], dim=2)

        print(f"shape of detections: {detections.shape}")

        return detections

    def process(self, images, visibility_models, return_time=False):
        with torch.no_grad():
            torch.cuda.synchronize()
            output = self.model(images)[-1]
            output['hm'] = output['hm'].sigmoid_()
            if self.opt.hm_hp and not self.opt.mse_loss:
                output['hm_hp'] = output['hm_hp'].sigmoid_()

            reg = output['reg'] if self.opt.reg_offset else None
            hm_hp = output['hm_hp'] if self.opt.hm_hp else None
            hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
            torch.cuda.synchronize()
            forward_time = time.time()

            dets = self.vehint_decode(
                images, visibility_models, output['hm'], output['wh'], output['hps'],
                reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)

            print(f"process dets shape: {dets.shape}")

        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def debug(self, debugger, images, dets, output, scale):
        dets = dets.detach().cpu().numpy().copy()
        dets[:, :, :4] *= self.opt.down_ratio
        dets[:, :, 5:53] *= self.opt.down_ratio
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip((img * 255.), 0, 255).astype(np.uint8)
        pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hm')
        if self.opt.hm_hp:
            pred = debugger.gen_colormap_hp(
                output['hm_hp'][0].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hmhp')

    def post_process(self, dets):
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        # print(f"input res: {self.opt.input_res}")
        dets = vehint_tl_post_process(dets.copy(), 2048)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 21)
            # import pdb; pdb.set_trace()
        return dets[0]

    def show_results(self, debugger, image, results):
        down_width = 512
        down_height = 512
        down_points = (down_width, down_height)
        print(f"image shape show results: {image.shape}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.show()
        for bbox in results[1]:
            if bbox[4] > self.opt.vis_thresh:
                bbox = np.array(bbox).astype(int)
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), thickness=3,
                              lineType=cv2.LINE_8)
                resize_down = cv2.resize(image, down_points, interpolation=cv2.INTER_LINEAR)
                keypoints = bbox[5:21].reshape(-1, 2)
                for i, keypoint in enumerate(keypoints):
                    cv2.circle(image, (keypoint[0], keypoint[1]), 3, self.colors[i], 2)
                    resize_down = cv2.resize(image, down_points, interpolation=cv2.INTER_LINEAR)

                for j, e in enumerate(self.edges):
                    if keypoints[e].min() > 0:
                        cv2.line(image, (keypoints[e[0], 0], keypoints[e[0], 1]),
                                 (keypoints[e[1], 0], keypoints[e[1], 1]), (255, 0, 0), 2,
                                 lineType=cv2.LINE_8)
                        resize_down = cv2.resize(image, down_points, interpolation=cv2.INTER_LINEAR)
        fig = plt.figure()
        fig.set_figwidth(20)
        fig.set_figheight(20)
        plt.imshow(image)
        plt.savefig('../data/data-apollocar3d/video_frames_pred/101.jpg')
        plt.show()

    def run(self, image_or_path_or_tensor, meta=None):
        # images_split/test/180310_024419662_Camera_5.jpg demo image for presentation
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug == 3),
                            theme=self.opt.debugger_theme)
        start_time = time.time()
        pre_processed = False
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type(''):
            image = cv2.imread(image_or_path_or_tensor)
        else:
            image = image_or_path_or_tensor['image'][0].numpy()
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True

        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        detections = []
        for scale in self.scales:
            scale_start_time = time.time()
            if not pre_processed:
                images = self.pre_process(image)
            else:
                # import pdb; pdb.set_trace()
                images = pre_processed_images['images'][scale][0]
            images = images.to(self.opt.device)
            torch.cuda.synchronize()
            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time

            vis_model_rll = models.resnet34(pretrained=True)
            vis_model_rll.fc = torch.nn.Sequential(
                torch.nn.Linear(in_features=512, out_features=1),
                torch.nn.Sigmoid()
            )
            path = "../exp/visibility_models/rear_L_light.pt"
            vis_model_rll.load_state_dict(torch.load(path))

            vis_model_rrl = models.resnet34(pretrained=True)
            vis_model_rrl.fc = torch.nn.Sequential(
                torch.nn.Linear(in_features=512, out_features=1),
                torch.nn.Sigmoid()
            )
            path = "../exp/visibility_models/rear_R_light.pt"
            vis_model_rrl.load_state_dict(torch.load(path))

            visibility_models = [
                vis_model_rll,
                vis_model_rrl
            ]

            output, dets, forward_time = self.process(images, visibility_models, return_time=True)

            torch.cuda.synchronize()
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time

            if self.opt.debug >= 2:
                self.debug(debugger, images, dets, output, scale)

            dets = self.post_process(dets)
            torch.cuda.synchronize()
            post_process_time = time.time()
            post_time += post_process_time - decode_time

            detections.append(dets)

        results = self.merge_outputs(detections)
        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time

        if self.opt.debug >= 1:
            images_viz = images.detach().cpu().numpy()
            images_viz = images_viz[0].transpose(1, 2, 0)
            self.show_results(debugger, images_viz, results)

        return {'results': results, 'tot': tot_time, 'load': load_time,
                'pre': pre_time, 'net': net_time, 'dec': dec_time,
                'post': post_time, 'merge': merge_time}
