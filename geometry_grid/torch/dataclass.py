from dataclasses import InitVar, asdict, dataclass, field, fields
from numbers import Number
import typing
from typing import Generic, Optional, TypeVar
from typeguard import typechecked
import typing_inspect


import torch

from py_structs.torch import shape
from py_structs import struct

from beartype import beartype


from jaxtyping import Float, Int, jaxtyped


from tensordict.prototype import tensorclass
from torch import Tensor
from jaxtyping.array_types import _FixedDim, _check_dims


def annotation_details(t):
  if hasattr(t, '__metadata__'):
    d = t.__metadata__[0]
    if '__torchtyping__' in d:
      return d['details']


N = TypeVar("N")

@dataclass(kw_only=True, repr=False)
class TensorClass(Generic[N]):
  # Broadcast prefixes shapes together
  broadcast:  InitVar[bool] = False   

  def __post_init__(self, broadcast):
    memo = {}
    variadic_memo = {}
    variadic_broadcast_memo = {}

    self.device = None
    for f in fields(self):
      value = getattr(self, f.name)
      
      if isinstance(value, torch.Tensor):
        f.type._check_shape(value, single_memo=memo, variadic_memo=variadic_memo, variadic_broadcast_memo=variadic_broadcast_memo)
        
      if isinstance(value, TensorClass):
        pass
        

    #     if annot is None or annot.shape is None:
    #       raise TypeError(f"Tensor field '{f.name}' must have a shape annotation")

    #     prefix[f.name], shapes[f.name] = check_shape(f.name, annot.shape, value)
    #     value = check_dtype(f.name, annot.dtype, value, convert_types)
    #     setattr(self, f.name, value)

    #     if self.device is not None and self.device != value.device:
    #       raise TypeError(f"Expected all tensors to have the same dtype, got {self.device} and {value.device}")
    #     self.device = value.device


    #   elif isinstance(value, TensorClass):
    #     prefix[f.name] = value.shape
    #     shapes[f.name] = value.shapes
    
    # if broadcast:
    #   try:
    #     prefix = torch.broadcast_shapes(*prefix.values())
    #     for k, sh in shapes.items():
    #       value = getattr(self, k)
    #       if isinstance(value, TensorClass):
    #         setattr(self, k, value.expand(self.prefix))
    #       else:
    #         setattr(self, k, value.expand(self.prefix + sh))

    #   except RuntimeError as e:
    #     raise TypeError(f"Could not broadcast shapes {prefix}") from e
    # else:
    #   prefixes = set(prefix.values())
    #   if len(prefixes) != 1:
    #     raise TypeError(
    #         f"Expected all tensors to have the same prefix, got: {prefix}")

    #   prefix = prefixes.pop()

    # self.shape = prefix
    # self.shapes = shapes

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

  @classmethod
  @typechecked
  def empty(cls:type, shape=(), device='cpu', **kwargs):
    if isinstance(shape, Number):
      shape = (shape,)
    
    def make_tensor(k, info):
      if k in kwargs:
        return kwargs[k]
      

      if info is None:
        raise RuntimeError(f"{k}: has no argument or tensor annotation")

      if info.dtype is None:
        raise RuntimeError(f"{k}: has no dtype annotation")
      
      field_shape = (d.size for d in info.shape.dims)
      return torch.empty( tuple( (*shape, *field_shape) ), 
        dtype=info.dtype.dtype, device=device)


    return cls(**{f.name:make_tensor(f.name, annot_info(f.type)) 
      for f in fields(cls)}) 




  def unsqueeze(self, dim):
    assert dim <= len(self.shape), f"Cannot unsqueeze dim {dim} in shape {self.shape}"
    return self.map(lambda t: t.unsqueeze(dim))


  def __repr__(self):
    name= self.__class__.__name__
    return f"{name}({shape(asdict(self))})"

@jaxtyped
@beartype
@dataclass
class T(TensorClass):
  a: Float[Tensor, "*#N 3"]
  b: Int[Tensor, "*#N 3"]
  c: str


@jaxtyped
@beartype
@dataclass
class F(TensorClass):
  a: T
  b: Int[Tensor, "*#N 3"]
  c: str


# T.__init__ = jaxtyped(T.__init__)

@jaxtyped
@beartype
def test_foo(a: Float[Tensor, "N 3"], b: Int[Tensor, "N 3"]):
  pass

if __name__ == "__main__":

  print(T.__init__.__annotations__)
  t = T(a=torch.randn(5, 1,3), b=torch.randn(5, 7,3).to(torch.long), c = "hello")
  f = F(a=t, b=torch.randn(1, 7,3).to(torch.long), c = "hello")

  # print(test_foo(t.a, t.b))
