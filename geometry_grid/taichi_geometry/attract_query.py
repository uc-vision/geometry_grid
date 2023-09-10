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
  query_radius: ti.f32

  force: vec3

  @ti.func
  def update(self, _, obj):
    d, p = obj.nearest_point(self.point)

    if d < self.query_radius and d > 1e-6:
      v = p - self.point 
      l = ti.math.length(v)
      
      self.force += self.sigma * (v/l) * ti.exp(- (l / self.sigma)**2) 





@ti.kernel
def _attract_query(object_grid:ti.template(), 
    points:ndarray(vec3, ndim=1), attenuation:ti.f32, query_radius:ti.f32,
    forces:ndarray(vec3, ndim=1)):
  
  for i in range(points.shape[0]):
    q = AttractQuery(points[i], attenuation, query_radius, vec3(0.))
    bounds = AABox(points[i] - query_radius, points[i] + query_radius)
    object_grid._query_grid(q, bounds)

    forces[i] = q.force


def attract_query (object_grid, points:torch.Tensor, 
                   sigma:ti.f32, query_radius:float) -> torch.FloatTensor:
  """ 
    Compute an attraction/repulsion force on each point due to the objects in the grid.
    force = exp(-(|v| / sigma)^2) * v / |v|

  """

  forces = torch.zeros((points.shape[0], 3), 
                       device=points.device, dtype=torch.float32)

  _attract_query(object_grid, points, sigma, query_radius, forces)
  return forces


