
import torch
from geometry_grid.taichi_geometry.conversion import converts_to

from geometry_grid.taichi_geometry.point_distances import (
  batch_distances, batch_distances_grad)
from .util import clear_grad

import taichi as ti
from tensorclass import TensorClass

from functools import cache

class BatchDistances(torch.autograd.Function):
  @staticmethod
  def forward(ctx, objects:TensorClass, points:torch.Tensor):

      distances = batch_distances(objects, points)

      ctx.save_for_backward(points, distances, *objects.tensors())
      ctx.object_type = type(objects)
      
      return distances

  @staticmethod
  def backward(ctx, grad_output):
      points, distances, *obj_tensors = ctx.saved_tensors
      objects = ctx.object_type.from_tensors(obj_tensors)

      return batch_distances_grad(objects, points, distances, grad_output)




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

  distances = BatchDistances.apply(segs, points)


  loss = distances.sum()
  loss.backward()

  print(loss, points.grad, segs.a.grad, segs.b.grad)