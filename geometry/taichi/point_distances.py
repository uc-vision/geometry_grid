import taichi as ti
import torch

from geometry.torch.random import random_segments

from ..torch.dataclass import TensorClass
from geometry.taichi.conversion import from_torch

@ti.kernel
def _min_distances(objects:ti.template(), 
  points:ti.types.ndarray(ti.math.vec3),
  radius:ti.f32,
  distances:ti.types.ndarray(ti.f32),
  indices:ti.types.ndarray(ti.i32)):

  for j in range(points.shape[0]):
    ti.loop_config(serialize=True)
    min_d = torch.inf
    index = -1

    for i in range(objects.shape[0]):
      d = objects[i].point_distance(points[j])
      if d < min_d and d < radius:
        min_d = d
        index = i


    distances[j] = min_d
    indices[j] = index
        

def min_distances(objects:TensorClass, points:torch.Tensor, max_distance:float=torch.inf):
  distances = torch.full((points.shape[0],), torch.inf, device=points.device, dtype=torch.float32)
  indexes = torch.full_like(distances, -1, dtype=torch.int32)

  objs = from_torch(objects)
  _min_distances(objs, points, max_distance, distances, indexes)
  return distances, indexes


@ti.kernel
def _distances(objects:ti.template(), 
  points:ti.types.ndarray(ti.math.vec3),
  distances:ti.types.ndarray(ti.f32)):

  for i in range(points.shape[0]):
    distances[i] = objects[i].point_distance(points[i])

def distances(objects:TensorClass, points:torch.Tensor):
  assert objects.shape[0] == points.shape[0]
  distances = torch.full((points.shape[0],), torch.inf, device=points.device, dtype=torch.float32)

  objs = from_torch(objects)
  _distances(objs, points, distances)
  return distances


if __name__ == "__main__":
  from geometry.torch.types import AABox

  x = random_segments(AABox(torch.tensor([-10.0, -10.0, -10.0]),
    torch.tensor([10.0, 10.0, 10.0])), n=10)

