from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .exdet import ExdetDetector
from .ddd import DddDetector
from .ctdet import CtdetDetector
from .multi_pose import MultiPoseDetector
from .base_detector_vehint import VehintDetector
from .vehint_kptreg_detector import VehintKptRegDetector
from .vehint_taillight_detector import VehintTaillightDetector
from .cascaded_model_detector import CascadedModelDetector

detector_factory = {
  'exdet': ExdetDetector, 
  'ddd': DddDetector,
  'ctdet': CtdetDetector,
  'multi_pose': MultiPoseDetector,
  'vehint': VehintDetector,
  'vehint_kptreg': VehintKptRegDetector,
  'vehint_taillight': VehintTaillightDetector,
  'offline_model1': VehintDetector,
  'cascaded': CascadedModelDetector
}
