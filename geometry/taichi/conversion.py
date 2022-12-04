from dataclasses import asdict, dataclass
import taichi as ti

import numpy as torch
from typeguard import typechecked

from geometry.torch.dataclass import TensorClass
from py_structs.torch import shape

numpy_taichi = {
    torch.float32: ti.f32,
    torch.float64: ti.f64,
    torch.int32: ti.i32,
    torch.int64: ti.i64,
    torch.int8: ti.i8,
    torch.int16: ti.i16,
    torch.uint8: ti.u8,
    torch.uint16: ti.u16,
}

taichi_torch = {v:k for k,v in numpy_taichi.items()}


@typechecked
def from_torch(x:torch.ndarray, dtype=None):
  if dtype is not None:
    x = x.astype(taichi_torch[dtype])
  else:
    assert x.dtype in numpy_taichi, f"Unsupported dtype {x.dtype}"
    dtype = numpy_taichi[x.dtype]

  v = ti.Vector.field(x.shape[-1], dtype=dtype, shape=x.shape[:-1])
  v.from_torch(x)
  return v

@typechecked
def torch_field(data, ti_struct:ti.lang.struct.StructType):
  
  field = ti_struct.field(shape=data.shape)
  field.from_torch(asdict(data))
  
  return field