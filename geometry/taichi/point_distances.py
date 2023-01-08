import taichi as ti
from taichi.types import ndarray
import torch

from geometry.torch.random import random_segments

from ..torch.dataclass import TensorClass
from geometry.taichi.conversion import from_torch

@ti.func 
def atomic_min_index(dist:ti.f32, index:ti.int32, 
    prev_dist:ti.f32, prev_index:ti.int32):
  old = ti.atomic_min(prev_dist, dist)
  if old != prev_dist:
      return dist, index
  else:
      return prev_dist, prev_index


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
  return distances, indices


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






@ti.data_oriented
def point_distance(objects:TensorClass, num_points:int):

  points = ti.Vector.field(3, dtype=ti.f32, shape=num_points)
  distances = ti.field(dtype=ti.f32, shape=num_points)
  indices = ti.field(dtype=ti.i32, shape=num_points)

  objects = from_torch(objects)

  @ti.kernel
  def _min_distances(n:ti.int32, max_radius:ti.f32):
    for j in range(n):
      distances[j], indices[j] = _min_distance(objects, points[j], max_radius)
    return distances, indices

  class DistanceFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_points:torch.Tensor, max_radius:float=torch.inf):

        out_dist = torch.empty(points.shape[0], device=points.device, dtype=torch.float32)
        points.from_torch(input_points)
    
        _min_distances(points.shape[0], torch.inf)
        distances.to_torch(out_dist)

        return out_dist

    @staticmethod
    def backward(ctx, outp_grad):
      ti.clear_all_gradients()
      distances.grad.from_torch(outp_grad)
      _min_distances.grad()

      inp_grad = torch.empty(points.shape[0], device=points.device, dtype=torch.float32)
      distances.grad.to_torch(inp_grad)
      
      return inp_grad

  def __call__(self, points:torch.Tensor):
    assert points.shape[0] <= self.points.shape[0]



  
                                                                                                                                                                                                                                                                                                                                                         



if __name__ == "__main__":
  from geometry.torch.types import AABox

  x = random_segments(AABox(torch.tensor([-10.0, -10.0, -10.0]),
    torch.tensor([10.0, 10.0, 10.0])), n=10)

