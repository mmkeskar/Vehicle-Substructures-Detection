from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

from models.decode import multi_pose_decode
from models.model import create_model, load_model
from utils.image import get_affine_transform
from utils.post_process import vehint_post_process
from utils.debugger import Debugger
from models.decode import _nms, _topk, _topk_channel
from models.utils import _gather_feat, _transpose_and_gather_feat

import matplotlib.pyplot as plt

class VehintDetector(object):
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
                      [4, 5], [4, 6], [5, 7], [6, 7],
                      [8, 9], [8, 10], [9, 11], [10, 11],
                      [12, 13], [12, 14], [13, 15], [14, 15],
                      [16, 17], [16, 19], [17, 18], [18, 19],
                      [20, 21], [20, 23], [21, 22], [22, 23]]

    def pre_process(self, image):
        input_res = 2048  # self.opt.input_res
        """
        image = cv2.resize(image, (input_res, input_res), interpolation=cv2.INTER_AREA)
        image = (image.astype(np.float32) / 255.)
        inp_height, inp_width = self.opt.input_h, self.opt.input_w
        images = image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)"""

        input_h = image.shape[0]
        input_w = image.shape[1]

        permissible_row = input_h - input_res  # 2710 - input_res
        permissible_col = input_w - input_res  # 3384 - input_res
        rand_row = np.random.randint(0, permissible_row) # 1680 #
        rand_col = np.random.randint(0, permissible_col) # 640 #

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

    def vehint_decode(self, hm, wh, kps, reg=None, hm_hp=None, hp_offset=None, K=10):
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
        detections = torch.cat([bboxes, scores, kps, clses], dim=2)

        return detections

    def process(self, images, return_time=False):
        with torch.no_grad():
            torch.cuda.synchronize()
            print(f"images.shape: {images.shape}")
            output = self.model(images)[-1]
            output['hm'] = output['hm'].sigmoid_()
            if self.opt.hm_hp and not self.opt.mse_loss:
                output['hm_hp'] = output['hm_hp'].sigmoid_()
            print(f"shape of hm: {output['hm'].shape}")

            reg = output['reg'] if self.opt.reg_offset else None
            hm_hp = output['hm_hp'] if self.opt.hm_hp else None
            hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
            torch.cuda.synchronize()
            forward_time = time.time()

            dets = self.vehint_decode(
                output['hm'], output['wh'], output['hps'],
                reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)

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
        dets = vehint_post_process(dets.copy(), 2048)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 53)
            # import pdb; pdb.set_trace()
        return dets[0]

    def show_results(self, debugger, image, results):
        down_width = 512
        down_height = 512
        down_points = (down_width, down_height)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for bbox in results[1]:
            if bbox[4] > self.opt.vis_thresh:
                bbox = np.array(bbox).astype(int)
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), thickness=3,
                              lineType=cv2.LINE_8)
                resize_down = cv2.resize(image, down_points, interpolation=cv2.INTER_LINEAR)
                keypoints = bbox[5:53].reshape(-1, 2)
                print(f"bbox: {bbox}")
                for keypoint in keypoints:
                    cv2.circle(image, (keypoint[0], keypoint[1]), 3, (0, 255, 0), 2)
                    resize_down = cv2.resize(image, down_points, interpolation=cv2.INTER_LINEAR)
                """
                for j, e in enumerate(self.edges):
                    if keypoints[e].min() > 0:
                        cv2.line(image, (keypoints[e[0], 0], keypoints[e[0], 1]),
                                 (keypoints[e[1], 0], keypoints[e[1], 1]), (255, 0, 0), 2,
                                 lineType=cv2.LINE_8)
                        resize_down = cv2.resize(image, down_points, interpolation=cv2.INTER_LINEAR)
                """
        plt.imshow(image)
        plt.show()

    def run(self, image_or_path_or_tensor, meta=None):
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

            output, dets, forward_time = self.process(images, return_time=True)

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
