import dataclasses
from typing import Tuple
import taichi as ti
from taichi.math import vec3
import numpy as np

from geometry import torch as torch_geom
from typeguard import typechecked

import torch
from geometry.taichi.types import AABox, Segment, Tube
from .conversion import from_torch, torch_field


@ti.dataclass
class Intersection:
  t : ti.f32
  i : ti.i32


ti.ScalarField

@ti.data_oriented
class BoxIntersection:
  def __init__(self, tubes:ti.Field, boxes:ti.Field, max_intersections=10):
    self.tubes = tubes  
    self.boxes = boxes 

    self.intersection = ti.field(ti.i32) # Intersection.field()
    box = ti.root.dense(ti.i, self.boxes.shape)
    hits = box.dynamic(ti.j, max_intersections, chunk_size=4)
    hits.place(self.intersection)

    self.n_box = ti.field(ti.i32, shape=self.boxes.shape)
    self.n_tube = ti.field(ti.i32, shape=self.tubes.shape)



  def from_torch(skeleton:torch_geom.Skeleton, boxes:AABox, max_intersections=10): 
    return BoxIntersection(
      tubes=torch_field(skeleton.tubes, Tube),
      boxes=torch_field(boxes, AABox),
      max_intersections=max_intersections)


  @ti.kernel
  def compute(self):

    for i in self.boxes:
      for j in range(self.tubes.shape[0]):

        if self.tubes[j].intersects_box(self.boxes[i]):
          self.intersection[i].append(j)
          
          self.n_box[i] += 1
          self.n_tube[j] += 1


@ti.func
def point_bounds(points:ti.template()) -> AABox:
  b = AABox(points[0], points[0])

  for i in points:
    b.lower = ti.math.min(b.lower, points[i])
    b.upper = ti.math.max(b.upper, points[i])
  return b


@ti.data_oriented
class Grid:
  def __init__(self, bounds:ti.Field, size:Tuple[int, int, int], max_intersections=10):

    self.bounds = bounds # AABox.field(shape=1)
    self.size = size

    self.intersection = ti.field(ti.i32) # Intersection.field()
    box = ti.root.pointer(ti.ijk, size)
    hits = box.dynamic(ti.l, max_intersections, chunk_size=4)
    hits.place(self.intersection)

  def from_torch(box:torch_geom.AABox, size:Tuple[int, int, int] | int,  max_intersections=10): 
    assert box.shape == ()

    return Grid(
      bounds=torch_field(box.unsqueeze(0), AABox),
      size = size if isinstance(size, tuple) else (size, size, size),
      max_intersections=max_intersections)

  @ti.kernel
  def intersect_tubes(self, tubes:ti.template()):
    b = self.bounds[0]
    extents = b.extents()
    d = [extents[i] / self.size[i] for i in range(3)]

    for i in range(self.size[0]):
      for j in range(self.size[1]):
        for k in range(self.size[2]):
          box = AABox(
            b.lower + vec3(i,j,k) * d,
            b.lower + vec3(i+1,j+1,k+1) * d)
            

          for l in range(tubes.shape[0]):
            if tubes[l].segment.intersects_box(box):
              self.intersection[i,j,k].append(l)


if __name__ == "__main__":
  ti.init(arch=ti.gpu)

  torch_geom.AABox(lower = torch_geom.tensor([0,0,0]), 
    upper = torch_geom.tensor([1,1,1]))

  
