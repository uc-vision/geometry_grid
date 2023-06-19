from functools import cache
import taichi as ti
from taichi.types import ndarray
import torch

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
def min_point_object(objects:TensorClass, points:torch.Tensor, max_distance:float=torch.inf):
  distances = torch.full((points.shape[0],), torch.inf, device=points.device, dtype=torch.float32)
  indexes = torch.full_like(distances, -1, dtype=torch.int32)

  k = min_distances_kernel(converts_to(objects))

  k(objects.flat(), points, max_distance, distances, indexes)
  return distances, indexes

@cache
def pairwise_distances_kernel(obj_struct):
  size = struct_size(obj_struct)

  @ti.kernel
  def k(objects:ti.types.ndarray(dtype=ti.types.vector(size, ti.f32), ndim=1),
    points:ndarray(ti.math.vec3), distances:ndarray(ti.f32)):
    for i in range(points.shape[0]):
      obj = from_vec(obj_struct, objects[i])
      distances[i] = obj.point_distance(points[i])
  
  return k


def pairwise_distances(objects:TensorClass, points:torch.Tensor):
  assert objects.batch_shape[0] == points.shape[0]
  distances = torch.full((points.shape[0],), torch.inf, device=points.device, dtype=torch.float32)

  k = pairwise_distances_kernel(converts_to(objects))

  k(objects.flat(), points, distances)
  return distances


def grad_like(t):
  return torch.zeros_like(t.grad) if t.grad is not None else None

def pairwise_distance_func(obj_struct):
  kernel =  pairwise_distances_kernel(obj_struct)

  class PointDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, objects:torch.Tensor, points:torch.Tensor):

        distances = torch.full((points.shape[0],), torch.inf, device=points.device, dtype=torch.float32)
        kernel(objects, points, distances)

        ctx.save_for_backward(objects, points, distances)

        return distances

    @staticmethod
    def backward(ctx, grad_output):
        objects, points, distances = ctx.saved_tensors

        distances.grad = grad_output.contiguous()
        kernel.grad(objects, points, distances)
        
        return grad_like(objects), grad_like(points)
    
  return PointDistance.apply

if __name__ == "__main__":
  ti.init(arch=ti.gpu, debug=True)

  import geometry_grid.torch_geometry.geometry_types as torch_geom
  import geometry_grid.taichi_geometry.geometry_types as ti_geom

  a = torch.randn(10, 3, requires_grad=True, device='cuda')

  segs = torch_geom.Segment(a, torch.randn(10, 3)).to(device='cuda')
  points = torch.randn(10, 3).to(device='cuda')

  segs = segs.requires_grad_(True)
  points.requires_grad = True


  distance_func = pairwise_distance_func(ti_geom.Segment)
  distances = distance_func(segs.flat(), points)

  loss = distances.sum()
  loss.backward()


  print(loss, points.grad, segs.a.grad)