
from geometry.taichi.point_distances import _min_distances
import taichi as ti
from taichi.types import ndarray
from taichi.math import vec3

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




def point_distance(objects:TensorClass):
  objects = from_torch(objects)


  class DistanceFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points:torch.Tensor):

        ctx.points = ndarray_vec3(points)
        ctx.distances = ti.ndarray(shape=points.shape[0], 
          dtype=ti.f32, needs_grad=points.requires_grad)

        ctx.indices = ti.ndarray(shape=points.shape[0], dtype=ti.int32)

        ctx.device = points.device    
        _min_distances(objects, ctx.points, torch.inf, ctx.distances, ctx.indices)

        distances = torch.empty(points.shape[0], device=points.device, dtype=torch.float32)
        _copy_f32(ctx.distances, distances)

        indices = torch.empty(points.shape[0], device=points.device, dtype=torch.int32)
        _copy_i32(ctx.indices, indices)

        return distances, indices

    @staticmethod
    def backward(ctx, outp_grad, _):
      ti.clear_all_gradients()

      _copy_f32(outp_grad.contiguous(), ctx.distances.grad)
      _min_distances.grad(objects, ctx.points, torch.inf, ctx.distances, ctx.indices)
      
      inp_grad = torch.empty(ctx.points.shape[0], device=ctx.device, dtype=torch.float32)
      _copy_vec3(ctx.points.grad, inp_grad)
      
      return inp_grad

  return DistanceFunc.apply




if __name__ == "__main__":
  ti.init(debug=True)

  from geometry.torch.random import random_segments, around_segments
  from geometry.torch.types import AABox

  segs = random_segments(AABox(torch.tensor([-5.,-5.,-5.]), 
    torch.tensor([5.,5.,5.])), n=50)
  points = around_segments(segs, 0.5, 1000)
  points.requires_grad = True

  dist_func = point_distance(segs)
  dist, indices = dist_func(points)

  err = dist.sum()
  err.backward()

  print(points.grad)
