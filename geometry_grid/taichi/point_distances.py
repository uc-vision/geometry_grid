import taichi as ti
from taichi.types import ndarray
import torch

from geometry_grid.torch.random import random_segments

from tensorclass import TensorClass
from geometry_grid.taichi.conversion import from_torch

from taichi.math import vec3

@ti.func 
def atomic_min_index(dist:ti.f32, index:ti.int32, 
    prev_dist:ti.f32, prev_index:ti.int32):
  old = ti.atomic_min(prev_dist, dist)
  if old == prev_dist:
    dist = prev_dist
    index = prev_index    
  return dist, index
  

@ti.func
def _min_distance(objects:ti.template(), 
  point:ti.math.vec3, radius:ti.f32):
    min_d = torch.inf
    index = -1

    for i in range(objects.shape[0]):
      d = objects[i].point_distance(point)
      if d < radius:
        min_d, index = atomic_min_index(d, i, min_d, index)

    return min_d, index



@ti.kernel
def _min_distances(objects:ti.template(), 
  points:ndarray(ti.math.vec3), radius:ti.f32,
  distances:ndarray(ti.f32), indices:ndarray(ti.i32)):

  for j in range(points.shape[0]):
    distances[j], indices[j] = _min_distance(objects, points[j], radius)


def min_distances(objects:TensorClass, points:torch.Tensor, max_distance:float=torch.inf):
  distances = torch.full((points.shape[0],), torch.inf, device=points.device, dtype=torch.float32)
  indexes = torch.full_like(distances, -1, dtype=torch.int32)

  objs = from_torch(objects)
  _min_distances(objs, points, max_distance, distances, indexes)
  return distances, indexes


@ti.kernel
def _distances(objects:ti.template(), 
  points:ndarray(ti.math.vec3), distances:ndarray(ti.f32)):
  for i in range(points.shape[0]):
    distances[i] = objects[i].point_distance(points[i])

def distances(objects:TensorClass, points:torch.Tensor):
  assert objects.shape[0] == points.shape[0]
  distances = torch.full((points.shape[0],), torch.inf, device=points.device, dtype=torch.float32)

  objs = from_torch(objects)
  _distances(objs, points, distances)
  return distances



  
                                                                                                                                                                                                                                                                                                                                                         



if __name__ == "__main__":
  from geometry_grid.torch.types import AABox

  x = random_segments(AABox(torch.tensor([-10.0, -10.0, -10.0]),
    torch.tensor([10.0, 10.0, 10.0])), n=10)

