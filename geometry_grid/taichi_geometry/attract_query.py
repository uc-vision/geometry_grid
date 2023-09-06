from typing import Tuple
import taichi as ti
from taichi.math import vec3, normalize
from taichi.types import ndarray
import torch

from .geometry_types import AABox


@ti.dataclass
class AttractQuery:
  point: vec3
  sigma: ti.f32
  max_distance: ti.f32

  force: vec3

  @ti.func
  def update(self, _, obj):
    d, p = obj.nearest_point(self.point)

    if d < self.max_distance and d > 1e-6:
      v = p - self.point 
      l = ti.math.length(v)
      
      self.force += v * ti.exp(- (l / self.sigma)**2) / l





@ti.kernel
def _attract_query(object_grid:ti.template(), 
    points:ndarray(vec3, ndim=1), attenuation:ti.f32, max_distance:ti.f32,
    forces:ndarray(vec3, ndim=1)):
  
  for i in range(points.shape[0]):
    q = AttractQuery(points[i], attenuation, max_distance, vec3(0.))
    bounds = AABox(points[i] - max_distance, points[i] + max_distance)
    object_grid._query_grid(q, bounds)

    forces[i] = q.force


def attract_query (object_grid, points:torch.Tensor, 
                   sigma:ti.f32, max_distance:float) -> torch.FloatTensor:
  """ 
    Compute an attraction/repulsion force on each point due to the objects in the grid.
    force = exp(-(|v| / sigma)^2) * v / |v|

  """

  forces = torch.zeros((points.shape[0], 3), 
                       device=points.device, dtype=torch.float32)

  _attract_query(object_grid, points, sigma, max_distance, forces)
  return forces


