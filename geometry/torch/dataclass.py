from dataclasses import asdict, dataclass
from torchtyping import patch_typeguard, TensorType
import torch

from os.path import commonprefix
from py_structs.torch import shape
from typeguard import typechecked

patch_typeguard()


class TensorClass:
  def __post__init__(self):
    shapes = [t.shape for t in self.__dict__.values() if isinstance(t, torch.Tensor)]
    self.shape = commonprefix(shapes)
        
  def map(self, func):
    d = {k:(func(t) if isinstance(t, torch.Tensor) else t) 
      for k, t in asdict(self)}

    return self.__class__(**d)

  def __getitem__(self, slice):
    return self.map(lambda t: t[slice])

  def to(self, device):
    return self.map(lambda t: t.to(device))

  def __repr__(self):
    name= self.__class__.__name__
    return f"{name}({shape(asdict(self))})"
