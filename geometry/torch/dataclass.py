from dataclasses import asdict, dataclass, fields
import typing
from torchtyping import patch_typeguard, ShapeDetail, DtypeDetail

import torch

from py_structs.torch import shape
from py_structs import struct


patch_typeguard()


def annotation_details(t):
  if hasattr(t, '__metadata__'):
    d = t.__metadata__[0]
    if '__torchtyping__' in d:
      return d['details']
  

def annot_info(t) -> typing.Optional[ShapeDetail]:
  details = annotation_details(t)
  shape, dtype = None, None
  if details:
    for d in details:
      if isinstance(d, ShapeDetail):
        shape = d
      elif isinstance(d, DtypeDetail):
        dtype = d

    return shape, dtype

      


def check_shape(name, shape, v):
  if not isinstance(v, torch.Tensor):
    raise TypeError(f'{name} must be a tensor, got {type(v)}')
  n = len(shape.dims)
  prefix, suffix = v.shape[:-n], v.shape[-n:]

  fake = struct(shape=v.shape[-n:], names=[None] * n)
  if not shape.check(fake):
    raise TypeError(f"Expected {name} to have shape {shape.dims}, got {v.shape}")
    
  return prefix, suffix
  
@dataclass(kw_only=True)
class TensorClass:
  broadcast: bool = False

  def __post_init__(self):
    prefix, shapes = {}, {}

    for f in fields(self):
      value = getattr(self, f.name)
      
      if isinstance(value, torch.Tensor):
        shape, dtype = annot_info(f.type)

        assert shape is not None, f"Tensor {f.name} must have a shape annotation"
        prefix[f.name], shapes[f.name] = check_shape(f.name, shape, value)
        if dtype is not None:
          if not dtype.check(value):
            raise TypeError(f"Expected {f.name} to have dtype {dtype.dtype}, got {value.dtype}")

      elif isinstance(value, TensorClass):
        prefix[f.name] = value.prefix
        shapes[f.name] = value.shapes


    self.shapes = shapes
    if self.broadcast:
      try:
        common = torch.broadcast_shapes(*prefix.values())
        for k, sh in shapes:
          value = getattr(self, k)
          if isinstance(value, TensorClass):
            setattr(self, k, value.expand(common))
          else:
            setattr(self, k, value.expand(common + sh))

      except RuntimeError as e:
        raise TypeError(f"Could not broadcast shapes {prefix}") from e
    else:
      prefixes = set(prefix.values())
      if len(prefixes) != 1:
        raise TypeError(
            f"Expected all tensors to have the same prefix, got: {prefix}")

      self.prefix = prefixes.pop()

      

  def map(self, func):
    d = {k:(func(t) if isinstance(t, torch.Tensor) else t) 
      for k, t in asdict(self)}

    return self.__class__(**d)

  def __getitem__(self, slice):
    return self.map(lambda t: t[slice])

  def to(self, device):
    return self.map(lambda t: t.to(device))

  def expand(self, shape):
    return self.map(lambda t: t.expand(shape))


  def __repr__(self):
    name= self.__class__.__name__
    return f"{name}({shape(asdict(self))})"
