from dataclasses import asdict, dataclass, is_dataclass
from typing import Mapping, Optional
import taichi as ti

import torch
from typeguard import typechecked

from geometry.torch.dataclass import TensorClass
from py_structs.torch import shape

import geometry.torch.types as torch_geom
import geometry.taichi.types as ti_geom

torch_taichi = {
    torch.float32: ti.f32,
    torch.float64: ti.f64,
    torch.int32: ti.i32,
    torch.int64: ti.i64,
    torch.int8: ti.i8,
    torch.int16: ti.i16,
    torch.uint8: ti.u8,
    # torch.uint16: ti.u16,
}



taichi_torch = {v:k for k,v in torch_taichi.items()}


@typechecked
def from_tensor(x:torch.Tensor, dtype=None):
  
  if dtype is not None:
    x = x.astype(taichi_torch[dtype])
  else:
    assert x.dtype in torch_taichi, f"Unsupported dtype {x.dtype}"
    dtype = torch_taichi[x.dtype]

  v = ti.Vector.field(x.shape[-1], dtype=dtype, shape=x.shape[:-1])
  v.from_torch(x)
  return v

def check_conversion(data:TensorClass, ti_struct:ti.lang.struct.StructType):
  data_name = data.__class__.__name__
  
  for k, v in ti_struct.members.items():
    if not hasattr(data, k):
      raise TypeError(f"Missing field in struct {k} in {data_name}")

    if isinstance(v, ti.lang.matrix.VectorType):
      if data.shapes[k] != v.get_shape():
        raise TypeError(f"Expected {k} to have shape {v.get_shape()}, got {data.shapes[v]}")
    elif isinstance(v, ti.lang.struct.StructType):
      check_conversion(getattr(data, k), v)


def taichi_shape(ti_type):
  if isinstance(ti_type, ti.lang.struct.StructType):
    return {k:taichi_shape(v) for k, v in ti_type.members.items()}

  if (isinstance(ti_type, ti.lang.matrix.VectorType) or 
      isinstance(ti_type, ti.lang.matrix.MatrixType)):
    return (ti_type.get_shape(), taichi_torch[ti_type.dtype])

  else:
    raise TypeError(f"Unsupported type {ti_type}")


torch_ti_mapping = {
  torch_geom.Sphere : ti_geom.Sphere,
  torch_geom.AABox : ti_geom.AABox,
  torch_geom.Segment : ti_geom.Segment,
  torch_geom.Line : ti_geom.Line,
  torch_geom.Tube : ti_geom.Tube
}

ti_torch_mapping = {v:k for k,v in torch_ti_mapping.items()}


def find_struct(tensor_class:type):
  assert tensor_class in torch_ti_mapping, f"Unsupported type {tensor_class}"
  return torch_ti_mapping[tensor_class]



@typechecked
def torch_field(data:TensorClass):
  ti_struct = find_struct(data.__class__)

  if data.shape_info != taichi_shape(ti_struct):
    raise TypeError(f"Expected shapes don't match:\n{data.shape_info}\n{taichi_shape(ti_struct)}")

  field = ti_struct.field(shape=data.shape)
  field.from_torch(asdict(data))
  
  return field




@typechecked
def from_torch(data:torch.Tensor | TensorClass):
  if isinstance(data, TensorClass):
    return torch_field(data)
  elif isinstance(data, TensorClass):
    return from_tensor(data)
  elif is_dataclass(data):
    return from_torch(asdict(data))
  elif isinstance(data, Mapping):
    return {k:from_torch(v) for k, v in data.items()}
  else:
    return data