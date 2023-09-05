import torch
from geometry_grid.taichi_geometry.conversion import converts_from, struct_size

import taichi as ti


@ti.kernel
def _get_objects_kernel(objects:ti.template(), indexes:ti.types.ndarray(ti.i32, ndim=1), obj_vecs:ti.types.ndarray(ndim=2)):
  for i in range(indexes.shape[0]):
    if indexes[i] >= 0:
      v = objects[indexes[i]].to_vec()
      for j in range(len(v)):
        obj_vecs[i, j] = v[j]


def get_object_vecs(grid, indexes:torch.Tensor):
    obj_vecs = torch.empty((indexes.shape[0], struct_size(grid.object_types)), 
      device=grid.device, dtype=torch.float32)

    _get_objects_kernel(grid.objects, indexes, obj_vecs)
    return obj_vecs
  
def get_objects(grid, indexes:torch.Tensor):
  obj_vecs = get_object_vecs(grid, indexes)
  tensorclass = converts_from(grid.object_types)
  return tensorclass.from_vec(obj_vecs)
  