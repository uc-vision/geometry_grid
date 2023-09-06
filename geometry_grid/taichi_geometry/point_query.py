from functools import cache
from typing import Tuple

import taichi as ti
from taichi.math import vec3
from taichi.types import ndarray
import torch

from .geometry_types import AABox


@ti.dataclass
class PointQuery:
  point: vec3
  query_radius: ti.f32
  distance: ti.f32
  index: ti.i32

  @ti.func
  def update(self, index, obj):
    d = distance_func(obj, self.point)
    if d < self.query_radius and ((d > 0.) or allow_zero):
      old = ti.atomic_min(self.distance, d)
      if old != self.distance:
        self.index = index




@ti.kernel
def _point_query(grid_index:ti.template(), 
    points:ndarray(vec3, ndim=1), 
    make_query:ti.template(),

    query_radius:ti.f32,
    distances:ndarray(ti.f32, ndim=1), 
    indexes:ndarray(ti.i32, ndim=1)):
  
  for i in range(points.shape[0]):
    bounds = AABox(points[i] - query_radius, points[i] + query_radius)
    q = make_query(points[i], query_radius)


    grid_index._query_grid(q, bounds)

    distances[i] = q.distance
    indexes[i] = q.index


def point_query (grid, points:torch.Tensor, query_radius:float,
                 allow_zero:bool=False, distance_func=None,
                 ) -> Tuple[torch.FloatTensor, torch.IntTensor]:

  distances = torch.empty((points.shape[0],), 
                          device=points.device, dtype=torch.float32)
  indexes = torch.empty_like(distances, dtype=torch.int32)


  make_query = _make_query(
    distance_func=distance_func or grid.object_types.methods['point_distance'],
    allow_zero=allow_zero)

  _point_query(grid.index, points, make_query, query_radius, distances, indexes)
  return distances, indexes


