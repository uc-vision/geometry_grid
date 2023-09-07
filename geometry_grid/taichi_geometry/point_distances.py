from functools import cache
from typing import Callable
from beartype import beartype
import taichi as ti
from taichi.types import ndarray
import torch

from geometry_grid.torch_geometry.random import random_segments

from geometry_grid.taichi_geometry.conversion import\
    check_conversion, converts_to, from_torch, struct_size, from_vec

from geometry_grid.torch_geometry.typecheck import typechecked
from tensorclass import TensorClass
from . import atomic



@ti.func
def _min_point_objects(point:ti.math.vec3,
                        obj_struct:ti.template(), obj_arr:ti.template()):
    index = atomic.pack_index(-1, torch.inf)

    for i in range(obj_arr.shape[0]):
      obj = from_vec(obj_struct, obj_arr[i])
      d = obj.point_distance(point)
      
      index2 = atomic.pack_index(i, d)
      ti.atomic_min(index, index2)

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
      indices[j], distances[j] = _min_point_objects(points[j], radius, obj_struct, obj_arr)

  return k

@typechecked
def min_point_distances(objects:TensorClass, points:torch.Tensor):
  distances = torch.full((points.shape[0],), torch.inf, device=points.device, dtype=torch.float32)
  indexes = torch.full_like(distances, -1, dtype=torch.int32)

  k = min_distances_kernel(converts_to(objects))

  k(objects.to_vec(), points, distances, indexes)
  return distances, indexes

@cache
def point_distance_obj(obj_struct):
  vec_type = ti.types.vector(struct_size(obj_struct), ti.f32)

  @ti.func
  def point_distance_obj(obj:vec_type, point:ti.math.vec3):
    obj = from_vec(obj_struct, obj)
    return obj.point_distance(point)
  
  return point_distance_obj

@cache
@beartype
def batch_distances_kernel(distance_func:Callable, n1:int, n2:int):

  @ti.kernel
  def k(vec1:ti.types.ndarray(dtype=ti.types.vector(n1, ti.f32), ndim=1),
        vec2:ti.types.ndarray(dtype=ti.types.vector(n2, ti.f32), ndim=1),
    distances:ndarray(ti.f32)):

    for i in range(vec1.shape[0]):
      distances[i] = distance_func(vec1, vec2)
  
  return k


def batch_distances(distance_func, obj1:torch.Tensor, obj2:torch.Tensor):
  assert obj1.shape[0] == obj2.shape[0]
  distances = torch.full((obj1.shape[0],), torch.inf, device=obj1.device, dtype=torch.float32)

  kernel = batch_distances_kernel(obj1.shape[1], obj2.shape[1], distance_func)
  kernel(obj1, obj2, distances)
  
  return distances



def batch_point_distances(objects:TensorClass, points:torch.Tensor):
  assert objects.batch_shape[0] == points.shape[0]
  distances = torch.full((points.shape[0],), torch.inf, device=points.device, dtype=torch.float32)
  obj_struct = converts_to(objects)

  kernel = point_distances_kernel(obj_struct)
  kernel(objects.to_vec(), points, distances)
  return distances

def point_distances_kernel(obj_struct):
    return batch_distances_kernel(point_distance_obj(obj_struct), struct_size(obj_struct), 3)
