
import torch
from geometry_grid.taichi_geometry.conversion import converts_to

from geometry_grid.taichi_geometry.point_distances import (
  point_distances_kernel)
from .util import clear_grad

import taichi as ti
from tensorclass import TensorClass

from functools import cache


def batch_point_distances(objects:TensorClass, points:torch.Tensor):
  assert objects.batch_shape == points.shape[:1], f"batch size does not match: {objects.batch_shape} != {points.shape[:1]}"

  struct_type = converts_to(objects)
  f = batch_distances_func(struct_type)
  return f(objects.to_vec(), points)

@cache
def batch_distances_func(obj_struct):
  kernel = point_distances_kernel(obj_struct)
  
  class BatchDistances(torch.autograd.Function):
    @staticmethod
    def forward(ctx, obj_vec:torch.Tensor, points:torch.Tensor):
        distances = torch.empty((points.shape[0],), dtype=torch.float32, device=points.device)
        kernel(obj_vec, points, distances)
        ctx.save_for_backward(points, distances, obj_vec)        
        return distances

    @staticmethod
    def backward(ctx, grad_output):
        points, distances, obj_vec = ctx.saved_tensors
        with clear_grad(obj_vec, points, distances):
          distances.grad = grad_output.contiguous()
          obj_vec.requires_grad_(True)

          kernel.grad(obj_vec, points, distances)
          return obj_vec.grad, points.grad
  return BatchDistances.apply

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

  distances = batch_point_distances(segs, points)

  loss = distances.sum()
  loss.backward()

  print(loss, points.grad, segs.a.grad, segs.b.grad)