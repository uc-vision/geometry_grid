
import torch

from geometry_grid.taichi_geometry.point_distances import (
  batch_distances_kernel)

import  geometry_grid.taichi_geometry.point_query as pq
from .util import clear_grad

from functools import cache


@cache
def point_query_func(grid, max_distance, allow_zero=False):
  kernel = batch_distances_kernel(grid.object_types)
  
  class BatchDistances(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points:torch.Tensor):
        distances, indexes = pq.point_query(grid, points, max_distance=max_distance, allow_zero=allow_zero)
        ctx.save_for_backward(points, distances)        
        return distances, indexes

    @staticmethod
    def backward(ctx, grad_output, indexes):
        points, distances = ctx.saved_tensors
        obj_vec = grid.get_object_vecs(indexes)

        with clear_grad(points, distances):
          distances.grad = grad_output.contiguous()
          kernel.grad(obj_vec, points, distances)

          return points.grad
  return BatchDistances.apply


def point_query(grid, points, max_distance, allow_zero=False):
  f = point_query_func(grid, max_distance, allow_zero=allow_zero)
  return f(points)