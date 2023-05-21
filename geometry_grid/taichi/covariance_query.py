from typing import Tuple
import taichi as ti
from taichi.math import vec3
from taichi.types import ndarray
import torch

from .types import AABox



@ti.dataclass
class CovarianceQuery:
  point : vec3
  radius : ti.f32

  mean: vec3
  cov: ti.math.mat3
  n : ti.i32

  @ti.func
  def update(self, index, other):
    d = other.point_distance(self.point)
    if d < self.radius:
      delta = other.point - self.mean
      self.mean += delta / (self.n + 1)
      self.cov += delta.outer_product(delta) * self.n / (self.n + 1)
      self.n += 1


  @ti.func
  def bounds(self) -> AABox:
    lower = self.point - self.radius
    upper = self.point + self.radius
    return AABox(lower, upper)

  @ti.func
  def basis(self) -> ti.math.mat3:
    vals, vecs = ti.math.sym_eig(self.cov)
    return vals * vecs


@ti.kernel
def _eig_query(object_grid:ti.template(), 
    points:ndarray(vec3, ndim=1), radius:ti.f32,
    bases:ndarray(ti.math.mat3, ndim=1),
    counts:ndarray(ti.i32, ndim=1)):
  
  for i in range(points.shape[0]):
    q = CovarianceQuery(points[i], radius, 
        mean = vec3(0,0,0), cov = ti.math.mat3(0), n = 0)
    object_grid._query_grid(q)
    result = q.result()

    bases[i] = result.basis
    counts[i] = result.n


def eig_query (object_grid, points:torch.Tensor, 
    radius:float) -> Tuple[torch.FloatTensor, torch.IntTensor]:
  
  bases = torch.empty((points.shape[0], 3, 3), device=points.device, dtype=torch.float32)
  counts = torch.empty((points.shape[0],), device=points.device, dtype=torch.int32)

  _eig_query(object_grid, points, radius, bases, counts)
  return bases, counts



