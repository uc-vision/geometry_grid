from typing import Tuple
from beartype import beartype
import taichi as ti
from taichi.types import ndarray
import torch

from geometry_grid.torch_geometry.typecheck import NFloat32, NInt32

from .geometry_types import AABox
from geometry_grid.taichi_geometry import atomic, conversion

from tensorclass import TensorClass

@beartype
def build_grid_points_query(point_grid, ti_struct:ti.lang.struct.StructType):
  n = conversion.struct_size(ti_struct)
  obj_vec = ti.types.vector(n, dtype=ti.f32)

  num_points = point_grid.objects.shape[0]
  packed_dist = ti.field(dtype=ti.u64, shape=(num_points,))

  @ti.dataclass 
  class QueryPoints:
    obj:ti_struct
    index:ti.i32
    max_distance:ti.f32

    @ti.func
    def update(self, index, point):
      d = self.obj.point_distance(point)
      
      if d < self.max_distance:
        packed = atomic.pack_index(self.index, d)
        ti.atomic_min(packed_dist[index], packed)
        
    @ti.func
    def bounds(self) -> AABox:
      b = self.obj.bounds() 
      return AABox(b.lower - self.max_distance, 
                   b.upper + self.max_distance)

  @ti.kernel
  def _query_points(objs:ndarray(obj_vec, ndim=1), 
      max_distance:ti.f32,
      distances:ndarray(ti.f32, ndim=1), 
      indexes:ndarray(ti.i32, ndim=1)): 
    
    for i in range(packed_dist.shape[0]):
      packed_dist[i] = atomic.pack_index(atomic.FloatIndex(-1, torch.inf))
    
    for i in range(objs.shape[0]):
      q = QueryPoints(obj=objs[i], index=i, max_distance=max_distance)
      point_grid._query_grid(q)

    for i in range(packed_dist.shape[0]):
      index = atomic.unpack_index(packed_dist[i])
      distances[i] = index.distance
      indexes[i] = index.index

  @beartype
  def f(objs:TensorClass, max_distance:float) -> Tuple[NInt32, NFloat32]:
    conversion.check_conversion(objs, ti_struct)

    distances = torch.empty((num_points,), device=objs.device, dtype=torch.float32)
    indexes = torch.empty_like(distances, dtype=torch.int32)
    
    _query_points(objs.flat(), max_distance, distances, indexes)

    return distances, indexes

  return f