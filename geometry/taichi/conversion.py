from dataclasses import dataclass
from typing import Any
import taichi as ti

import numpy as np

from .types import GridBounds
import geometry.np as np_geom

from nptyping import NDArray, Shape
from typeguard import typechecked

numpy_taichi = {
    np.float32: ti.f32,
    np.float64: ti.f64,
    np.int32: ti.i32,
    np.int64: ti.i64,
    np.int8: ti.i8,
    np.int16: ti.i16,
    np.uint8: ti.u8,
    np.uint16: ti.u16,
}

taichi_numpy = {v:k for k,v in numpy_taichi.items()}


@typechecked
def from_numpy(x:np.ndarray, dtype:ti.DataType=None):
  if dtype is not None:
    x = x.astype(taichi_numpy[dtype])
  else:
    dtype = numpy_taichi[x.dtype]

  v = ti.Vector.field(x.shape[-1], dtype=dtype, shape=x.shape[:-1])
  v.from_numpy(x)
  return v
