
from geometry.taichi.point_distances import _min_distances
import taichi as ti
from taichi.types import ndarray
from taichi.math import vec3

import geometry.torch as torch_geom

import torch

from geometry.torch.random import random_segments

from geometry.torch.dataclass import TensorClass
from geometry.taichi.conversion import from_torch



@ti.kernel
def _copy_vec3(src:ndarray(vec3), dest:ndarray(vec3)):
  for i in range(src.shape[0]):
    dest[i] = src[i]

def ndarray_vec3(src:torch.Tensor):
  dest = ti.ndarray(shape=src.shape[0], dtype=vec3, needs_grad=src.requires_grad)
  _copy_vec3(src, dest)
  return dest


@ti.kernel
def _copy_f32(src:ndarray(ti.f32), dest:ndarray(ti.f32)):
  for i in range(src.shape[0]):
    dest[i] = src[i]

@ti.kernel
def _copy_i32(src:ndarray(ti.int32), dest:ndarray(ti.int32)):
  for i in range(src.shape[0]):
    dest[i] = src[i]

def ndarray_f32(src:torch.Tensor):
  dest = ti.ndarray(shape=src.shape[0], dtype=ti.f32)
  _copy_f32(src, dest)
  return dest


@ti.kernel
def _load_segments(f:ti.template(), 
  a:ti.types.ndarray(dtype=vec3, ndim=1), 
  b:ti.types.ndarray(dtype=vec3, ndim=1)):
  
  for i in range(f.shape[0]):
    f[i] = torch_geom.Segment(a[i], b[i])

def load_segments(segs:torch_geom.Segment):
  f = torch_geom.Segment.field(shape=segs.shape[0])
  _load_segments(f, segs.a, segs.b)
  return f


