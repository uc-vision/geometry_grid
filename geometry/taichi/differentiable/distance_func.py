
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
  dest = ti.ndarray(src.shape[0], vec3)
  _copy_vec3(src, dest)
  return dest



def point_distance(objects:TensorClass):
  objects = from_torch(objects)


  class DistanceFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points:torch.Tensor):

        ctx.inputs = ndarray_vec3(points)
        ctx.distances = ti.ndarray(points.shape[0], ti.f32)

        ctx.device = points.device
        indices = torch.empty(points.shape[0], device=points.device, dtype=torch.float32)
    
        _min_distances(ctx.input, torch.inf, distances, indices)

        distances = torch.empty(points.shape[0], device=points.device, dtype=torch.float32)
        _copy_vec3(ctx.distances, distances)

        return distances, indices

    @staticmethod
    def backward(ctx, outp_grad):
      ti.clear_all_gradients()

      ctx.distances.grad.from_torch(outp_grad)
      _min_distances.grad()
      
      inp_grad = torch.empty(ctx.inputs.shape[0], device=ctx.device, dtype=torch.float32)
      _copy_vec3(ctx.inputs.grad, inp_grad)
      
      return inp_grad

  return DistanceFunc.apply




if __name__ == "__main__":

  segs = 