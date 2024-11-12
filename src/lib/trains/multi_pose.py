from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torchvision import models
import numpy as np
import copy

from models.losses import FocalLoss, RegL1Loss, RegLoss, RegWeightedL1Loss, KeypointVisLoss
from models.decode import multi_pose_decode, vehint_decode, offline_model1_decode, offline_model2_decode
from models.utils import _sigmoid, flip_tensor, flip_lr_off, flip_lr
from utils.debugger import Debugger
from utils.post_process import multi_pose_post_process, vehint_post_process, offline_model1_post_process, \
  vehint_tl_post_process, vehint_kptreg_post_process, cascaded_model_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer

class MultiPoseLoss(torch.nn.Module):
  def __init__(self, opt):
    super(MultiPoseLoss, self).__init__()
    self.crit = FocalLoss()
    self.crit_hm_hp = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_kp = RegWeightedL1Loss() if not opt.dense_hp else \
                   torch.nn.L1Loss(reduction='sum')
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
                    RegLoss() if opt.reg_loss == 'sl1' else None
    self.opt = opt
    self.vis_loss = KeypointVisLoss()

  def forward(self, outputs, batch):
    opt = self.opt
    hm_loss, wh_loss, off_loss = 0, 0, 0
    hp_loss, off_loss, hm_hp_loss, hp_offset_loss, kps_vis_loss = 0, 0, 0, 0, 0
    for s in range(opt.num_stacks):
      output = outputs[s]
      output['hm'] = _sigmoid(output['hm'])    # the function clips all the values
      if opt.hm_hp and not opt.mse_loss:
        output['hm_hp'] = _sigmoid(output['hm_hp'])
      
      if opt.eval_oracle_hmhp:
        output['hm_hp'] = batch['hm_hp']
      if opt.eval_oracle_hm:
        output['hm'] = batch['hm']
      if opt.eval_oracle_kps:
        if opt.dense_hp:
          output['hps'] = batch['dense_hps']
        else:
          output['hps'] = torch.from_numpy(gen_oracle_map(
            batch['hps'].detach().cpu().numpy(), 
            batch['ind'].detach().cpu().numpy(), 
            opt.output_res, opt.output_res)).to(opt.device)
      if opt.eval_oracle_hp_offset:
        output['hp_offset'] = torch.from_numpy(gen_oracle_map(
          batch['hp_offset'].detach().cpu().numpy(), 
          batch['hp_ind'].detach().cpu().numpy(), 
          opt.output_res, opt.output_res)).to(opt.device)

      hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks   # self.crit is the FocalLoss
      if opt.hp_weight > 0:
        if opt.dense_hp:
          mask_weight = batch['dense_hps_mask'].sum() + 1e-4
          hp_loss += (self.crit_kp(output['hps'] * batch['dense_hps_mask'],
                                   batch['dense_hps'] * batch['dense_hps_mask']) /
                                   mask_weight) / opt.num_stacks
        else:
          hp_loss += self.crit_kp(output['hps'], batch['hps_mask'],
                                  batch['ind'], batch['hps']) / opt.num_stacks
      if opt.wh_weight > 0:
        wh_loss += self.crit_reg(output['wh'], batch['reg_mask'],
                                 batch['ind'], batch['wh']) / opt.num_stacks
      if opt.reg_offset and opt.off_weight > 0:
        off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                  batch['ind'], batch['reg']) / opt.num_stacks
      if opt.reg_hp_offset and opt.off_weight > 0:
        hp_offset_loss += self.crit_reg(
          output['hp_offset'], batch['hp_mask'],
          batch['hp_ind'], batch['hp_offset']) / opt.num_stacks
      if opt.hm_hp and opt.hm_hp_weight > 0:
        hm_hp_loss += self.crit_hm_hp(
          output['hm_hp'], batch['hm_hp']) / opt.num_stacks

    loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
           opt.off_weight * off_loss + opt.hp_weight * hp_loss + \
           opt.hm_hp_weight * hm_hp_loss + opt.off_weight * hp_offset_loss
    
    loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'hp_loss': hp_loss, 
                  'hm_hp_loss': hm_hp_loss, 'hp_offset_loss': hp_offset_loss,
                  'wh_loss': wh_loss, 'off_loss': off_loss}
    return loss, loss_stats

class MultiPoseTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(MultiPoseTrainer, self).__init__(opt, model, optimizer=optimizer)
    if self.opt.vis_models:
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

      self.visibility_models = [
        vis_model_fll,
        vis_model_rll,
        vis_model_rrl,
        vis_model_frl,
        vis_model_fp,
        vis_model_rp
      ]

    if opt.task == 'cascaded':
      """
      kpt_reg_model_lf = models.resnet50(pretrained=True)
      kpt_reg_model_lf.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features=kpt_reg_model_lf.fc.in_features, out_features=8),
        torch.nn.Tanh()
      )
      path = "../exp/regression_model/Left_Front.pt"
      kpt_reg_model_lf.load_state_dict(copy.deepcopy((torch.load(path))))

      kpt_reg_model_lr = models.resnet50(pretrained=True)
      kpt_reg_model_lr.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features=kpt_reg_model_lr.fc.in_features, out_features=8),
        torch.nn.Tanh()
      )
      path = "../exp/regression_model/Left_Rear.pt"
      kpt_reg_model_lr.load_state_dict(copy.deepcopy((torch.load(path))))

      kpt_reg_model_rf = models.resnet50(pretrained=True)
      kpt_reg_model_rf.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features=kpt_reg_model_rf.fc.in_features, out_features=8),
        torch.nn.Tanh()
      )
      path = "../exp/regression_model/Right_Front.pt"
      kpt_reg_model_rf.load_state_dict(copy.deepcopy((torch.load(path))))

      kpt_reg_model_rr = models.resnet50(pretrained=True)
      kpt_reg_model_rr.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features=kpt_reg_model_rr.fc.in_features, out_features=8),
        torch.nn.Tanh()
      )
      path = "../exp/regression_model/Right_Rear.pt"
      kpt_reg_model_rr.load_state_dict(copy.deepcopy((torch.load(path))))
      """
      kpt_reg_model_lf = torch.load("../exp/regression_model/Left_Front.pt")
      kpt_reg_model_lr = torch.load("../exp/regression_model/Left_Rear.pt")
      kpt_reg_model_rf = torch.load("../exp/regression_model/Right_Front.pt")
      kpt_reg_model_rr = torch.load("../exp/regression_model/Right_Rear.pt")

      self.regression_models = [
        kpt_reg_model_lf,
        kpt_reg_model_lr,
        kpt_reg_model_rr,
        kpt_reg_model_rf
      ]

  def _get_losses(self, opt):
    loss_states = ['loss', 'hm_loss', 'hp_loss', 'hm_hp_loss', 
                   'hp_offset_loss', 'wh_loss', 'off_loss']
    loss = MultiPoseLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None
    hm_hp = output['hm_hp'] if opt.hm_hp else None
    hp_offset = output['hp_offset'] if opt.reg_hp_offset else None
    dets = multi_pose_decode(
      output['hm'], output['wh'], output['hps'], 
      reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])

    dets[:, :, :4] *= opt.input_res / opt.output_res
    dets[:, :, 5:39] *= opt.input_res / opt.output_res
    dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    dets_gt[:, :, :4] *= opt.input_res / opt.output_res
    dets_gt[:, :, 5:39] *= opt.input_res / opt.output_res
    for i in range(1):
      debugger = Debugger(
        dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')

      debugger.add_img(img, img_id='out_pred')
      for k in range(len(dets[i])):
        if dets[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                 dets[i, k, 4], img_id='out_pred')
          debugger.add_coco_hp(dets[i, k, 5:39], img_id='out_pred')

      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                 dets_gt[i, k, 4], img_id='out_gt')
          debugger.add_coco_hp(dets_gt[i, k, 5:39], img_id='out_gt')

      if opt.hm_hp:
        pred = debugger.gen_colormap_hp(output['hm_hp'][i].detach().cpu().numpy())
        gt = debugger.gen_colormap_hp(batch['hm_hp'][i].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hmhp')
        debugger.add_blend_img(img, gt, 'gt_hmhp')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    reg = output['reg'] if self.opt.reg_offset else None
    hm_hp = output['hm_hp'] if self.opt.hm_hp else None
    hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
    if self.opt.task == 'vehint':
      if self.opt.kpts16_vehint:
        kpts16 = True
      else:
        kpts16 = False
      if self.opt.vis_models:
        dets = vehint_decode(batch['input'],
          output['hm'], output['wh'], output['hps'], visibility_models=self.visibility_models,
          reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K, kpts16=kpts16)
      else:
        dets = vehint_decode(batch['input'],
          output['hm'], output['wh'], output['hps'],
          reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K, kpts16=kpts16)
    elif self.opt.task == 'offline_model1':
      if self.opt.vis_models:
        dets = offline_model1_decode(batch['input'],
                                     output['hm'], output['wh'], output['hps'],
                                     visibility_models=self.visibility_models,
                                     reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)
      else:
        dets = offline_model1_decode(batch['input'],
                                     output['hm'], output['wh'], output['hps'],
                                     reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)
    elif self.opt.task == 'cascaded':
      if self.opt.vis_models:
        dets = offline_model1_decode(batch['input'],
                                     output['hm'], output['wh'], output['hps'],
                                     visibility_models=self.visibility_models,
                                     reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)
      else:
        dets = offline_model1_decode(batch['input'],
                                     output['hm'], output['wh'], output['hps'],
                                     reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)
      dets = offline_model2_decode(batch["input"], self.regression_models, dets)
    else:
      dets = multi_pose_decode(
        output['hm'], output['wh'], output['hps'],
        reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)
    """
    dets = vehint_decode(batch['input'], self.visibility_models,
      output['hm'], output['wh'], output['hps'],
      reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)
    """
    """
    dets = multi_pose_decode(
      output['hm'], output['wh'], output['hps'],
      reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)
    """
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])

    if 'c' in batch['meta'].keys():
      dets_out = multi_pose_post_process(
        dets.copy(), batch['meta']['c'].cpu().numpy(),
        batch['meta']['s'].cpu().numpy(),
        output['hm'].shape[2], output['hm'].shape[3])
    elif self.opt.task == 'vehint':
      dets_out = vehint_post_process(dets.copy(), self.opt.input_res, kpts16=kpts16)
    elif self.opt.task == 'vehint_taillight':
      dets_out = vehint_tl_post_process(dets.copy(), self.opt.input_res)
    elif self.opt.task == 'offline_model1':
      dets_out = offline_model1_post_process(dets.copy(), self.opt.input_res)
    elif self.opt.task == 'cascaded':
      dets_out = cascaded_model_post_process(dets.copy(), self.opt.input_res)
    elif self.opt.task == 'vehint_kptreg':
      dets_out = vehint_kptreg_post_process(dets.copy(), self.opt.input_res)
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
