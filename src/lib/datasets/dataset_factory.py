from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.vehint import VehIntDataset
from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.multi_pose import MultiPoseDataset
from .sample.vehint2 import VehIntDataset2
from .sample.vehint_kptreg import VehIntKptRegDataset

from .dataset.apollo3d import Apollo3d
from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP


dataset_factory = {
  'apollo': Apollo3d,
  'coco': COCO,
  'pascal': PascalVOC,
  'kitti': KITTI,
  'coco_hp': COCOHP
}

_sample_factory = {
  'vehint': VehIntDataset,
  'exdet': EXDetDataset,
  'ctdet': CTDetDataset,
  'ddd': DddDataset,
  'multi_pose': MultiPoseDataset,
  'vehint2': VehIntDataset2,
  'vehint_kptreg': VehIntKptRegDataset
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
