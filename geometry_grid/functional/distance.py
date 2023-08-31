
import torch
from geometry_grid.taichi_geometry.conversion import converts_to

from geometry_grid.taichi_geometry.point_distances import (
  batch_distances_kernel, min_distances_kernel)
from .util import clear_grad

import taichi as ti
from tensorclass import TensorClass

from functools import cache

@cache
def batch_distances_func(obj_struct):
  kernel =  batch_distances_kernel(obj_struct)
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

        with clear_grad(objects, points, distances):
          distances.grad = grad_output.contiguous()
          kernel.grad(objects, points, distances)
          return objects.grad, points.grad
  return PointDistance.apply





def batch_distances(objects:TensorClass, points:torch.Tensor):
  f = batch_distances_func(converts_to(objects))
  return f(objects.flat(), points)

if __name__ == "__main__":
  ti.init(arch=ti.gpu, debug=True)


  import geometry_grid.torch_geometry.geometry_types as torch_geom
  import geometry_grid.taichi_geometry.geometry_types as ti_geom

  torch.cuda.manual_seed(0)

  a = torch.randn(10, 3, requires_grad=True, device='cuda')
  b = torch.randn(10, 3, requires_grad=True, device='cuda')

  segs = torch_geom.Segment(a, b).to(device='cuda')
  points = torch.randn(10, 3).to(device='cuda')

  segs = segs.requires_grad_(True)
  points.requires_grad = True

  distances = batch_distances(segs, points)


  loss = distances.sum()
  loss.backward()

  print(loss, points.grad, segs.a.grad, segs.b.grad)