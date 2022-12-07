from dataclasses import InitVar, asdict, dataclass, field, fields
import typing
from typing import Optional
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
  
@dataclass
class TensorAnnotation:
  shape: ShapeDetail
  dtype: Optional[DtypeDetail]



def annot_info(t) -> Optional[TensorAnnotation]:
  details = annotation_details(t)
  dtype = None

  if details is not None:
    shape = ShapeDetail(dims=[], check_names=False)

    for d in details:
      if isinstance(d, ShapeDetail):
        shape = d
      elif isinstance(d, DtypeDetail):
        dtype = d

    return TensorAnnotation(shape, dtype)

      

def check_shape(name:str, sd:ShapeDetail, v:torch.Tensor):

  n = len(sd.dims)
  prefix, suffix = v.shape[:len(v.shape) - n], v.shape[len(v.shape) - n:]

  fake = struct(shape=v.shape[-n:], names=[None] * n)
  if not sd.check(fake):
    raise TypeError(f"Expected {name} to have shape {sd.dims}, got {v.shape}")
    
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
class TensorClass():
  # Broadcast prefixes shapes together
  broadcast:  InitVar[bool] = False   

  # Convert to annotated datatypes rather than throw TypeError
  convert_types: InitVar[bool] = False  


  def __post_init__(self, broadcast, convert_types):
    prefix, shapes = {}, {}

    for f in fields(self):
      value = getattr(self, f.name)
      
      if isinstance(value, torch.Tensor):
        annot = annot_info(f.type)

        if annot is None or annot.shape is None:
          raise TypeError(f"Tensor field '{f.name}' must have a shape annotation")

        prefix[f.name], shapes[f.name] = check_shape(f.name, annot.shape, value)
        value = check_dtype(f.name, annot.dtype, value, convert_types)
        setattr(self, f.name, value)

      elif isinstance(value, TensorClass):
        prefix[f.name] = value.shape
        shapes[f.name] = value.shapes
    
    if broadcast:
      try:
        prefix = torch.broadcast_shapes(*prefix.values())
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

      prefix = prefixes.pop()

    self.shape = prefix
    self.shapes = shapes

  @property 
  def shape_info(self):

    def info(k):
      v = getattr(self, k)
      if isinstance(v, TensorClass):
        return v.shape_info
      if isinstance(v, torch.Tensor):
        return (self.shapes[k], v.dtype)

    return {f.name: info(f.name)  
      for f in fields(self)
    }


  def __iter__(self):
    fs = fields(self)
    for f in fs:
      yield (f.name, getattr(self, f.name))

  def map(self, func):
    def f(t):
      if isinstance(t, torch.Tensor):
        return func(t)
      elif isinstance(t, TensorClass):
        return t.map(func)
      else:
        return t

    d = {k:f(t) for k, t in iter(self)}
    return self.__class__(**d)

  def __getitem__(self, slice):
    return self.map(lambda t: t[slice])

  def to(self, device):
    return self.map(lambda t: t.to(device))

  def expand(self, shape):
    return self.map(lambda t: t.expand(shape))


  def unsqueeze(self, dim):
    assert dim <= len(self.shape), f"Cannot unsqueeze dim {dim} in shape {self.shape}"
    return self.map(lambda t: t.unsqueeze(dim))


  def __repr__(self):
    name= self.__class__.__name__
    return f"{name}({shape(asdict(self))})"
