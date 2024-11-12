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

from models.decode import offline_model1_decode, offline_model2_decode
from models.model import create_model, load_model
from utils.image import get_affine_transform
from utils.post_process import cascaded_model_post_process
from utils.debugger import Debugger
from models.decode import _nms, _topk, _topk_channel
from models.utils import _gather_feat, _transpose_and_gather_feat

import matplotlib.pyplot as plt

class CascadedModelDetector(object):
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

        self.colors = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255),
                       (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),
                       (255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0),
                       (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255),
                       (255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255),
                       (255, 255, 0), (255, 255, 0), (255, 255, 0), (255, 255, 0)]

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

    def process(self, images, visibility_models, regression_models, return_time=False):
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

            # dets = offline_model1_decode(images,
            #                              output['hm'], output['wh'], output['hps'],
            #                              reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)
            # dets = offline_model2_decode(images, regression_models, dets)
            if self.opt.vis_models:
                dets = offline_model1_decode(images,
                                             output['hm'], output['wh'], output['hps'],
                                             visibility_models=visibility_models,
                                             reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)
            else:
                dets = offline_model1_decode(images,
                                             output['hm'], output['wh'], output['hps'],
                                             reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)
            dets = offline_model2_decode(images, regression_models, dets)



            print(f"process dets shape: {dets.shape}")

        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def debug(self, debugger, images, dets, output, scale):
        dets = dets.detach().cpu().numpy().copy()
        dets[:, :, :4] *= self.opt.down_ratio
        dets[:, :, 5:37] *= self.opt.down_ratio
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
        dets = cascaded_model_post_process(dets.copy(), self.opt.input_res)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 46)
            # import pdb; pdb.set_trace()
        return dets[0]

    def show_results(self, debugger, image, results):
        down_width = 512
        down_height = 512
        down_points = (down_width, down_height)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.show()
        for bbox in results[1]:
            if bbox[4] > self.opt.vis_thresh:
                bbox = np.array(bbox).astype(int)
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), thickness=3,
                              lineType=cv2.LINE_8)
                resize_down = cv2.resize(image, down_points, interpolation=cv2.INTER_LINEAR)
                lf = bbox[5:13].reshape(-1, 2)
                rf = bbox[29:37].reshape(-1, 2)
                keypoints = np.vstack((lf, rf))
                for i, keypoint in enumerate(keypoints):
                    print(i)
                    cv2.circle(image, (keypoint[0], keypoint[1]), 3, (0, 255, 0), 2)
                    resize_down = cv2.resize(image, down_points, interpolation=cv2.INTER_LINEAR)
                cv2.line(image, (keypoints[0][0], keypoints[0][1]),
                         (keypoints[1][0], keypoints[1][1]), (255, 0, 0), 2,
                         lineType=cv2.LINE_8)
                cv2.line(image, (keypoints[1][0], keypoints[1][1]),
                         (keypoints[3][0], keypoints[3][1]), (255, 0, 0), 2,
                         lineType=cv2.LINE_8)
                cv2.line(image, (keypoints[2][0], keypoints[2][1]),
                         (keypoints[3][0], keypoints[3][1]), (255, 0, 0), 2,
                         lineType=cv2.LINE_8)
                cv2.line(image, (keypoints[0][0], keypoints[0][1]),
                         (keypoints[2][0], keypoints[2][1]), (255, 0, 0), 2,
                         lineType=cv2.LINE_8)
                cv2.line(image, (keypoints[4][0], keypoints[4][1]),
                         (keypoints[5][0], keypoints[5][1]), (255, 0, 0), 2,
                         lineType=cv2.LINE_8)
                cv2.line(image, (keypoints[5][0], keypoints[5][1]),
                         (keypoints[7][0], keypoints[7][1]), (255, 0, 0), 2,
                         lineType=cv2.LINE_8)
                cv2.line(image, (keypoints[6][0], keypoints[6][1]),
                         (keypoints[7][0], keypoints[7][1]), (255, 0, 0), 2,
                         lineType=cv2.LINE_8)
                cv2.line(image, (keypoints[6][0], keypoints[6][1]),
                         (keypoints[4][0], keypoints[4][1]), (255, 0, 0), 2,
                         lineType=cv2.LINE_8)
                centers_lf = bbox[37:39].reshape(-1, 2)
                centers_rf = bbox[43:45].reshape(-1, 2)
                centers = np.vstack((centers_lf, centers_rf))
                for i, center in enumerate(centers):
                    cv2.circle(image, (center[0], center[1]), 3, (0, 0, 255), 2)
                    resize_down = cv2.resize(image, down_points, interpolation=cv2.INTER_LINEAR)
        fig = plt.figure()
        fig.set_figwidth(20)
        fig.set_figheight(20)
        plt.imshow(image)
        # plt.savefig('../data/data-apollocar3d/video_frames_pred/101.jpg')
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

            vis_model_fll = models.resnet34(pretrained=True)
            vis_model_fll.fc = torch.nn.Sequential(
                torch.nn.Linear(in_features=512, out_features=1),
                torch.nn.Sigmoid()
            )
            path = "../exp/visibility_models/front_L_light.pt"
            vis_model_fll.load_state_dict(torch.load(path))

            vis_model_frl = models.resnet34(pretrained=True)
            vis_model_frl.fc = torch.nn.Sequential(
                torch.nn.Linear(in_features=512, out_features=1),
                torch.nn.Sigmoid()
            )
            path = "../exp/visibility_models/front_R_light.pt"
            vis_model_frl.load_state_dict(torch.load(path))

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

            vis_model_fp = models.resnet34(pretrained=True)
            vis_model_fp.fc = torch.nn.Sequential(
                torch.nn.Linear(in_features=512, out_features=1),
                torch.nn.Sigmoid()
            )
            path = "../exp/visibility_models/front_plate.pt"
            vis_model_fp.load_state_dict(torch.load(path))

            vis_model_rp = models.resnet34(pretrained=True)
            vis_model_rp.fc = torch.nn.Sequential(
                torch.nn.Linear(in_features=512, out_features=1),
                torch.nn.Sigmoid()
            )
            path = "../exp/visibility_models/rear_plate.pt"
            vis_model_rp.load_state_dict(torch.load(path))

            """
            visibility_models = [
                vis_model_fll,
                vis_model_rll,
                vis_model_rrl,
                vis_model_frl,
                vis_model_fp,
                vis_model_rp
            ]
            """
            visibility_models = [
                vis_model_fll,
                vis_model_rll,
                vis_model_rrl,
                vis_model_frl
            ]

            kpt_reg_model_lf = torch.load("../exp/regression_model/Left_Front.pt")
            kpt_reg_model_lr = torch.load("../exp/regression_model/Left_Rear.pt")
            kpt_reg_model_rf = torch.load("../exp/regression_model/Right_Front.pt")
            kpt_reg_model_rr = torch.load("../exp/regression_model/Right_Rear.pt")

            regression_models = [
                kpt_reg_model_lf,
                kpt_reg_model_lr,
                kpt_reg_model_rr,
                kpt_reg_model_rf
            ]

            output, dets, forward_time = self.process(images, visibility_models,
                regression_models, return_time=True)

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
