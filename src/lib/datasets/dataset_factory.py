from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.vehint import VehIntDataset
from .sample.vehint_kptreg import VehIntKptRegDataset
from .sample.offline_model1 import VehInt6KptRegDataset

from .dataset.apollo3d import Apollo3d


dataset_factory = {
  'apollo': Apollo3d
}

_sample_factory = {
  'vehint': VehIntDataset,
  'vehint_kptreg': VehIntKptRegDataset,
  'offline_model1': VehInt6KptRegDataset,
  'cascaded': VehInt6KptRegDataset
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
