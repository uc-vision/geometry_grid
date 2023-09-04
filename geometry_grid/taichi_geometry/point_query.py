from typing import Tuple
import taichi as ti
from taichi.math import vec3
from taichi.types import ndarray
import torch

from .geometry_types import AABox


@ti.dataclass
class PointQuery:
  point: vec3
  max_distance: ti.f32

  distance: ti.f32
  index: ti.i32
  allow_zero: bool

  @ti.func
  def update(self, index, obj):
    d = obj.point_distance(self.point)
    if d < self.max_distance and ((d > 0.) or self.allow_zero):
      old = ti.atomic_min(self.distance, d)
      if old != self.distance:
        self.index = index


  @ti.func
  def bounds(self) -> AABox:
    lower = self.point - self.max_distance
    upper = self.point + self.max_distance
    return AABox(lower, upper)



@ti.kernel
def _point_query(grid_index:ti.template(), 
    points:ndarray(vec3, ndim=1), 
    max_distance:ti.f32,
    distances:ndarray(ti.f32, ndim=1), 
    indexes:ndarray(ti.i32, ndim=1),
    allow_zero:bool):
  
  for i in range(points.shape[0]):
    q = PointQuery(points[i], max_distance, 
                   distance=torch.inf, index=-1, allow_zero=allow_zero)
    grid_index._query_grid(q)

    distances[i] = q.distance
    indexes[i] = q.index


def point_query (grid, points:torch.Tensor, max_distance:float,
                 allow_zero:bool=False) -> Tuple[torch.FloatTensor, torch.IntTensor]:

  distances = torch.empty((points.shape[0],), 
                          device=points.device, dtype=torch.float32)
  indexes = torch.empty_like(distances, dtype=torch.int32)

  _point_query(grid.index, points, max_distance, distances, indexes, allow_zero)
  return distances, indexes


