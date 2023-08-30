from typing import Tuple
import taichi as ti
from taichi.math import vec3
from taichi.types import ndarray
import torch

from .geometry_types import AABox


@ti.dataclass
class AttractQuery:
  point: vec3
  attenuation: ti.f32
  max_distance: ti.f32

  force: vec3

  @ti.func
  def update(self, _, obj):
    d, p = obj.nearest_point(self.point)

    if d < self.max_distance and d > 0.:
      v = p - self.point 
      self.force += v * (self.attenuation / d)**2



  @ti.func
  def bounds(self) -> AABox:
    lower = self.point - self.max_distance
    upper = self.point + self.max_distance
    return AABox(lower, upper)



@ti.kernel
def _attract_query(object_grid:ti.template(), 
    points:ndarray(vec3, ndim=1), attenuation:ti.f32, max_distance:ti.f32,
    forces:ndarray(vec3, ndim=1)):
  
  for i in range(points.shape[0]):
    q = AttractQuery(points[i], attenuation, max_distance, vec3(0.))
    object_grid._query_grid(q)

    forces[i] = q.force


def attract_query (object_grid, points:torch.Tensor, attenuation:ti.f32, max_distance:float) -> torch.FloatTensor:
  """ 
    Compute an attraction/repulsion force on each point due to the objects in the grid.
    force = sum(  (attenuation / |v|)^2 *  v )

  """

  forces = torch.zeros((points.shape[0], 3), device=points.device, dtype=torch.float32)

  _attract_query(object_grid, points, attenuation, max_distance, forces)
  return forces


