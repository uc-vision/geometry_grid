from dataclasses import InitVar, asdict, dataclass, field, fields
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
  
def check_dtype(name, dtype, v, convert=False):
  if dtype is not None:
    if not dtype.check(v):
      if convert:
        return v.to(dtype.dtype)
      else:
        raise TypeError(f"Expected {name} to have dtype {dtype.dtype}, got {v.dtype}")

  return v

@dataclass(kw_only=True, repr=False)
class TensorClass:
  # Broadcast prefixes shapes together
  broadcast:  InitVar[bool] = False   

  # Convert to annotated datatypes rather than throw TypeError
  convert_types: InitVar[bool] = False  

  def __post_init__(self, broadcast, convert_types):
    prefix, shapes = {}, {}

    for f in fields(self):
      value = getattr(self, f.name)
      
      if isinstance(value, torch.Tensor):
        shape, dtype = annot_info(f.type)

        assert shape is not None, f"Tensor {f.name} must have a shape annotation"
        prefix[f.name], shapes[f.name] = check_shape(f.name, shape, value)
        value = check_dtype(f.name, dtype, value, convert_types)

      elif isinstance(value, TensorClass):
        prefix[f.name] = value.prefix
        shapes[f.name] = value.shapes


    self.shapes = shapes
    if broadcast:
      try:
        self.prefix = torch.broadcast_shapes(*prefix.values())
        for k, sh in shapes.items():
          value = getattr(self, k)
          if isinstance(value, TensorClass):
            setattr(self, k, value.expand(self.prefix))
          else:
            setattr(self, k, value.expand(self.prefix + sh))

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
