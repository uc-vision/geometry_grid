from functools import cache
import taichi as ti
from taichi.types import ndarray
import torch
from geometry_grid.functional.util import clear_grad

from geometry_grid.torch_geometry.random import random_segments

from geometry_grid.taichi_geometry.conversion import\
    check_conversion, converts_to, from_torch, struct_size, from_vec

from taichi.math import vec3

from geometry_grid.torch_geometry.typecheck import typechecked
from tensorclass import TensorClass
from . import atomic



@ti.func
def _min_point_objects(point:ti.math.vec3, radius:ti.f32,
                        obj_struct:ti.template(), obj_arr:ti.template()):
    index = atomic.unpack_index(-1, torch.inf)

    for i in range(obj_arr.shape[0]):
      obj = from_vec(obj_struct, obj_arr[i])
      d = obj.point_distance(point)
      
      if d < radius:
        atomic.atomic_min(index, d, i)

    return atomic.unpack_index(index)

def flat_type(obj_struct, objects):
  check_conversion(objects, obj_struct)
  size = struct_size(obj_struct)
  return ti.types.ndarray(dtype=ti.types.vector(size, ti.f32), ndim=1)

@cache
def min_distances_kernel(obj_struct):
  size = struct_size(obj_struct)

  @ti.kernel
  def k(obj_arr:ti.types.ndarray(dtype=ti.types.vector(size, ti.f32), ndim=1),
    points:ndarray(ti.math.vec3), radius:ti.f32,
    distances:ndarray(ti.f32), indices:ndarray(ti.i32)):

    for j in range(points.shape[0]):
      distances[j], indices[j] = _min_point_objects(points[j], radius, obj_struct, obj_arr)

  return k

@typechecked
def min_distances(objects:TensorClass, points:torch.Tensor, max_distance:float=torch.inf):
  distances = torch.full((points.shape[0],), torch.inf, device=points.device, dtype=torch.float32)
  indexes = torch.full_like(distances, -1, dtype=torch.int32)

  k = min_distances_kernel(converts_to(objects))

  k(objects.to_vec(), points, max_distance, distances, indexes)
  return distances, indexes

@cache
def batch_distances_kernel(obj_struct):
  size = struct_size(obj_struct)

  @ti.kernel
  def k(objects:ti.types.ndarray(dtype=ti.types.vector(size, ti.f32), ndim=1),
    points:ndarray(ti.math.vec3), distances:ndarray(ti.f32)):
    for i in range(points.shape[0]):
      obj = from_vec(obj_struct, objects[i])
      distances[i] = obj.point_distance(points[i])
  
  return k


def batch_distances(objects:TensorClass, points:torch.Tensor):
  assert objects.batch_shape[0] == points.shape[0]
  distances = torch.full((points.shape[0],), torch.inf, device=points.device, dtype=torch.float32)

  k = batch_distances_kernel(converts_to(objects))

  k(objects.to_vec(), points, distances)
  return distances


def batch_distances_grad(objects:TensorClass, points:torch.Tensor, distances:torch.Tensor, distances_grad:torch.Tensor):
  assert objects.batch_shape[0] == points.shape[0]

  k = batch_distances_kernel(converts_to(objects))
  obj_vec = objects.to_vec()

  with clear_grad(obj_vec, points, distances):
    distances.grad = distances_grad.contiguous()
    k.grad(obj_vec, points, distances)

    return objects.from_vec(obj_vec.grad), points.grad
