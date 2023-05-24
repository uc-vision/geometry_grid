from typing import Tuple
import taichi as ti
from taichi.math import vec3
from taichi.types import ndarray
import torch

from .geometry_types import AABox


@ti.dataclass
class PointQuery:
  point: vec3
  radius: ti.f32

  distance: ti.f32
  index: ti.i32
  comparisons: ti.i32

  @ti.func
  def update(self, index, other):
    d = other.point_distance(self.point)
    if d < self.radius:
      old = ti.atomic_min(self.distance, d)
      if old != self.distance:
        self.index = index


  @ti.func
  def bounds(self) -> AABox:
    lower = self.point - self.radius
    upper = self.point + self.radius
    return AABox(lower, upper)



@ti.kernel
def _point_query(object_grid:ti.template(), 
    points:ndarray(vec3, ndim=1), radius:ti.f32,
    distances:ndarray(ti.f32, ndim=1), indexes:ndarray(ti.i32, ndim=1)):
  
  for i in range(points.shape[0]):
    q = PointQuery(points[i], radius, distance=torch.inf, index=-1)
    object_grid._query_grid(q)

    distances[i] = q.distance
    indexes[i] = q.index


def point_query (object_grid, points:torch.Tensor, 
    radius:float) -> Tuple[torch.FloatTensor, torch.IntTensor]:

  distances = torch.empty((points.shape[0],), device=points.device, dtype=torch.float32)
  indexes = torch.empty_like(distances, dtype=torch.int32)

  _point_query(object_grid, points, radius, distances, indexes)
  return distances, indexes


