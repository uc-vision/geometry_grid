
import torch

from geometry_grid.taichi_geometry.point_distances import (
  batch_distances_kernel)

import  geometry_grid.taichi_geometry.point_query as pq
from geometry_grid.taichi_geometry.query_grid import get_object_vecs
from .util import clear_grad

from functools import cache


@cache
def point_query_func(grid, allow_zero=False, distance_func=None):

  kernel = batch_distances_kernel(grid.object_types, 
    distance_func = distance_func or grid.object_types.methods['point_distance'])
  
  class BatchDistances(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points:torch.Tensor):
        distances, indexes = pq.point_query(grid, points, allow_zero=allow_zero, distance_func=distance_func)
        ctx.save_for_backward(points, distances, indexes)       

        return distances, indexes

    @staticmethod
    def backward(ctx, grad_output, _):
        points, distances, indexes = ctx.saved_tensors
        obj_vec = get_object_vecs(grid, indexes)

        with clear_grad(points, distances):
          distances.grad = grad_output.contiguous()
          kernel.grad(obj_vec, points, distances)

          points.grad[indexes < 0] = 0.0

          return points.grad, None
  return BatchDistances.apply


def point_query(grid, points, query_radius, allow_zero=False, distance_func=None):
  f = point_query_func(grid, allow_zero=allow_zero, distance_func=distance_func)
  return f(points, query_radius)