from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetTrainer
from .ddd import DddTrainer
from .exdet import ExdetTrainer
from .multi_pose import MultiPoseTrainer
from .vehint_bbpred import VehIntTrainer
from .vehint_kptreg import VehintKptRegTrainer

train_factory = {
  'exdet': ExdetTrainer, 
  'ddd': DddTrainer,
  'ctdet': CtdetTrainer,
  'multi_pose': MultiPoseTrainer, 
  'vehint': MultiPoseTrainer,
  'vehint_bbpred': VehIntTrainer,
  'vehint_kptreg': VehintKptRegTrainer,
  'offline_model1': MultiPoseTrainer,
  'cascaded': MultiPoseTrainer
}
