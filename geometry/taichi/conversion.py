from dataclasses import asdict
import taichi as ti

import numpy as np
from typeguard import typechecked

_numpy_taichi = {
    np.float32: ti.f32,
    np.float64: ti.f64,
    np.int32: ti.i32,
    np.int64: ti.i64,
    np.int8: ti.i8,
    np.int16: ti.i16,
    np.uint8: ti.u8,
    np.uint16: ti.u16,
}

numpy_taichi = {np.dtype(k):v for k,v in _numpy_taichi.items()}
taichi_numpy = {v:k for k,v in numpy_taichi.items()}


@typechecked
def from_numpy(x:np.ndarray, dtype=None):
  if dtype is not None:
    x = x.astype(taichi_numpy[dtype])
  else:
    assert x.dtype in numpy_taichi, f"Unsupported dtype {x.dtype}"
    dtype = numpy_taichi[x.dtype]

  v = ti.Vector.field(x.shape[-1], dtype=dtype, shape=x.shape[:-1])
  v.from_numpy(x)
  return v


def to_field(data, ti_struct):
  
  field = ti_struct.field(shape=data.size)
  field.from_numpy(asdict(data))
  return field