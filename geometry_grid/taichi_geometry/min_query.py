from functools import cache
from typing import Tuple

import taichi as ti
from taichi.math import vec3
from taichi.types import ndarray
import torch

from geometry_grid.taichi_geometry.conversion import from_vec, struct_size

from .geometry_types import AABox, Point




@cache
def min_query_kernel(n:int, distance_func):
  vec_type = ti.types.vector(n, ti.f32)
  
  @ti.dataclass
  class MinQuery:
    obj: vec_type
    query_radius: ti.f32
    distance: ti.f32
    index: ti.i32

    @ti.func
    def update(self, index, obj):
      d = distance_func(obj, self.obj, self.query_radius)
      if d < self.distance:
        self.distance = d
        self.index = index


  @ti.kernel
  def kernel(grid_index:ti.template(), 
      points:ndarray(vec_type, ndim=1), 

      query_radius:ti.f32,
      distances:ndarray(ti.f32, ndim=1), 
      indexes:ndarray(ti.i32, ndim=1)):
    
    for i in range(points.shape[0]):
      bounds = AABox(points[i] - query_radius, points[i] + query_radius)
      q = MinQuery(points[i], query_radius, torch.inf, -1)

      grid_index._query_grid(q, bounds)

      distances[i] = q.distance
      indexes[i] = q.index

  return kernel





def min_query (grid, points:torch.Tensor, query_radius:float, distance_func) -> Tuple[torch.FloatTensor, torch.IntTensor]:

  distances = torch.empty((points.shape[0],), 
                          device=points.device, dtype=torch.float32)
  indexes = torch.empty_like(distances, dtype=torch.int32)

  query = min_query_kernel(points.shape[1], distance_func)
  query(grid.index, points, query_radius, distances, indexes)

  return distances, indexes


@ti.func 
def point_distance(obj:ti.template(), point:ti.math.vec3, query_radius:ti.f32):
  d = obj.point_distance(point)
  return ti.select(d <= query_radius, d, torch.inf)

def point_query (grid, points:torch.Tensor, query_radius:float) -> Tuple[torch.FloatTensor, torch.IntTensor]:
  return min_query(grid, points, query_radius, point_distance)


