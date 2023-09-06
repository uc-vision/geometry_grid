from functools import cache
from typing import Tuple

import taichi as ti
from taichi.math import vec3
from taichi.types import ndarray
import torch

from geometry_grid.taichi_geometry.conversion import from_vec, struct_size

from .geometry_types import AABox



@cache
def make_query(obj_type):
  vec_type = ti.types.vector(struct_size(obj_type), ti.f32)

  @ti.dataclass
  class MinQuery:
    item: vec_type
    query_radius: ti.f32
    distance: ti.f32
    index: ti.i32

    @ti.func
    def update(self, index, obj):
      d = self.item.distance(obj, self.query_radius)
      if d < self.distance:
        self.distance = d
        self.index = index


  @ti.kernel
  def query_kernel(grid_index:ti.template(), 
      points:ndarray(vec_type, ndim=1), 

      query_radius:ti.f32,
      distances:ndarray(ti.f32, ndim=1), 
      indexes:ndarray(ti.i32, ndim=1)):
    
    for i in range(points.shape[0]):
      bounds = AABox(points[i] - query_radius, points[i] + query_radius)
      q = MinQuery(from_vec(obj_type, points[i]), query_radius, torch.inf, -1)

      grid_index._query_grid(q, bounds)

      distances[i] = q.distance
      indexes[i] = q.index


  return query_kernel



def min_query (grid, points:torch.Tensor, query_radius:float,
                 obj_type:ti.lang.struct.StructType,
                 ) -> Tuple[torch.FloatTensor, torch.IntTensor]:

  distances = torch.empty((points.shape[0],), 
                          device=points.device, dtype=torch.float32)
  indexes = torch.empty_like(distances, dtype=torch.int32)


  query = make_query(obj_type)
  query(grid.index, points, query_radius, distances, indexes)

  return distances, indexes


