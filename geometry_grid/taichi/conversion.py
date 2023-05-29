from dataclasses import asdict, dataclass, is_dataclass
from functools import cache
from typing import Dict, Mapping, Optional, Sequence
from numpy import product
import taichi as ti

import torch
from beartype import beartype

from tensorclass import TensorClass
from py_structs.torch import shape

import geometry_grid.taichi.geometry_types as ti_geom
import geometry_grid.torch.geometry_types as torch_geom

from geometry_grid.torch.typecheck import typechecked



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


@beartype
def from_tensor(x:torch.Tensor, dtype=None):
  
  if dtype is not None:
    x = x.astype(taichi_torch[dtype])
  else:
    assert x.dtype in torch_taichi, f"Unsupported dtype {x.dtype}"
    dtype = torch_taichi[x.dtype]

  v = ti.Vector.field(x.shape[-1], dtype=dtype, shape=x.shape[:-1])
  v.from_torch(x)
  return v



def _check_shape(data_name, struct_shape, ti_struct:ti.lang.struct.StructType):
  for k, v in ti_struct.members.items():
    if not k in struct_shape:
      raise TypeError(f"Missing field in struct {k} in {data_name} {sorted(ti_struct.members.keys())} vs. {sorted(struct_shape.keys())}")

    if isinstance(v, ti.lang.matrix.VectorType):
      shape, dtype = struct_shape[k]

      if shape != v.get_shape():
        raise TypeError(f"Expected {k} to have shape {v.get_shape()}, got {shape}")
      
      if dtype != taichi_torch[v.dtype]:
        raise TypeError(f"Expected {k} to have dtype {taichi_torch[v.dtype]}, got {dtype}")

    elif isinstance(v, ti.lang.struct.StructType):
      _check_shape(f"{data_name}.{k}", struct_shape[k], v)


@typechecked
def check_static_conversion(cls:type, ti_struct:ti.lang.struct.StructType):
  static_shape = cls.static_shape_info()
  _check_shape(cls.__name__, static_shape, ti_struct)

@typechecked
def check_conversion(data:TensorClass, ti_struct:ti.lang.struct.StructType):
  _check_shape(data.__class__.__name__, data.shape_info, ti_struct)


def taichi_shape(ti_type):
  if isinstance(ti_type, ti.lang.struct.StructType):
    return {k:taichi_shape(v) for k, v in ti_type.members.items()}

  if (isinstance(ti_type, ti.lang.matrix.VectorType) or 
      isinstance(ti_type, ti.lang.matrix.MatrixType)):
    return (ti_type.get_shape(), taichi_torch[ti_type.dtype])

  else:
    return (1, taichi_torch[ti_type.dtype])



def struct_size(ti_struct:ti.lang.struct.StructType):
  size = 0
  for k, v in ti_struct.members.items():
    if isinstance(v, ti.lang.matrix.VectorType):

      size += product(v.get_shape())
    elif isinstance(v, ti.lang.struct.StructType):
      size += struct_size(v)
    else:
      size += 1
  return size



@typechecked
def tensorclass_field(data:TensorClass, dtype:ti.lang.struct.StructType):
  if data.shape_info != taichi_shape(dtype):
    raise TypeError(f"Expected shapes don't match:\n{data.shape_info}\n{taichi_shape(dtype)}")

  field = dtype.field(shape=data.batch_shape)
  field.from_torch(asdict(data))
  
  return field


def flatten_dicts(d:dict):
  out = []
  for k in sorted(d.keys()):
    v = d[k]

    if isinstance(v, dict):
      for k2, v2 in flatten_dicts(v):
        out.append( ( f"{k}.{k2}",  v2) )
    else:
      out.append( (k, v) )
  return out

def flatten_values(d:dict):
  out = []
  for k in sorted(d.keys()):
    v = d[k]
    if isinstance(v, dict):
      out.extend(flatten_values(v))
    else:
      out.append(v)
  return out


def type_str(shape, dtype):
    dtype_str = f"ti.{torch_taichi[dtype]})"
    if shape == 1:
      return torch_taichi[dtype]
    elif isinstance(shape, tuple):
      if len(shape) == 1:
        return f"ti.types.vector({shape[0]}, {dtype_str}"
      else:
        return f"ti.types.matrix({shape[0]}, {shape[1]}, {dtype_str}"
    raise TypeError(f"Unsupported shape {shape}")
    

_conversions = {
  torch_geom.AABox : ti_geom.AABox,
  torch_geom.Sphere : ti_geom.Sphere,
  torch_geom.Plane : ti_geom.Plane,
  torch_geom.Line : ti_geom.Line,
  torch_geom.Segment : ti_geom.Segment,
  torch_geom.Tube : ti_geom.Tube
}

conversions = {}   

def register_conversion(torch_type:type, ti_type:ti.lang.struct.StructType):
  assert issubclass(torch_type, TensorClass)
  check_static_conversion(torch_type, ti_type)

  conversions[torch_type] = ti_type

for k, v in _conversions.items():
  register_conversion(k, v)


def list_conversions():
  return [cls.__name__ for cls in conversions.keys()]

@typechecked
def converts_to(data:TensorClass):
  assert data.__class__ in conversions, f"Unsupported type {data.__class__}, options are {list_conversions()} use register_conversion"
  return conversions[data.__class__]


@typechecked
def from_torch(data:torch.Tensor | Mapping | Sequence | TensorClass):
  if isinstance(data, TensorClass):
    assert data.__class__ in conversions, f"Unsupported type {data.__class__},  options are {list_conversions()}, use register_conversion"
    return tensorclass_field(data, conversions[data.__class__])
  
  elif isinstance(data, torch.Tensor):
    return from_tensor(data)
  elif is_dataclass(data):
    return from_torch(asdict(data))
  elif isinstance(data, Mapping):
    return {k:from_torch(v) for k, v in data.items()}
  elif isinstance(data, Sequence):
    return [from_torch(v) for v in data]
  
  raise TypeError(f"Unsupported type {type(data)}")