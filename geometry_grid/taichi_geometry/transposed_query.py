from typing import Tuple
from beartype import beartype
import taichi as ti
from taichi.types import ndarray
import torch

from geometry_grid.torch_geometry.typecheck import NFloat32, NInt32

from .geometry_types import AABox, Point
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
    def update(self, index:ti.i32, point:Point):
      d = self.obj.point_distance(point.p)
      
      if d < self.max_distance:
        p = atomic.pack_index(self.index, d)
        ti.atomic_min(packed_dist[index], p)
        

  @ti.kernel
  def _query_points(objs:ndarray(obj_vec, ndim=1), 
      max_distance:ti.f32,
      distances:ndarray(ti.f32, ndim=1), 
      indexes:ndarray(ti.i32, ndim=1)): 
    
    for i in range(packed_dist.shape[0]):
      packed_dist[i] = atomic.pack_index(-1, torch.inf)
    
    for i in range(objs.shape[0]):
      obj = conversion.from_vec(ti_struct, objs[i])

      q = QueryPoints(obj=obj, index=i, max_distance=max_distance)
      b = obj.bounds() 
      bounds = AABox(b.lower - max_distance, 
                   b.upper + max_distance)

      point_grid._query_grid(q, bounds)

    for i in range(packed_dist.shape[0]):
      indexes[i], distances[i] = atomic.unpack_index(packed_dist[i])


  @beartype
  def f(objs:TensorClass, max_distance:float) -> Tuple[NFloat32, NInt32]:
    conversion.check_conversion(objs, ti_struct)

    distances = torch.empty((num_points,), device=objs.device, dtype=torch.float32)
    indexes = torch.empty_like(distances, dtype=torch.int32)
    
    _query_points(objs.to_vec(), max_distance, distances, indexes)
    return (distances, indexes)

  return f