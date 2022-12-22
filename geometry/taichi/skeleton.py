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
    self.box:ti.SNode = ti.root.bitmasked(ti.ijk, size)
    hits = self.box.dynamic(ti.l, max_intersections, chunk_size=4)
    hits.place(self.intersection)


  def from_torch(box:torch_geom.AABox, size:Tuple[int, int, int] | int,  max_intersections=60): 
    assert box.shape == ()

    return Grid(
      bounds=torch_field(box.unsqueeze(0), AABox),
      size = size if isinstance(size, tuple) else (size, size, size),
      max_intersections=max_intersections)

  @ti.func
  def get_inc(self) -> Tuple[vec3, vec3]:
    b = self.bounds[0]
    extents = b.extents()

    return b.lower, extents / vec3(self.size)


  @ti.kernel
  def intersect_dense(self, objects:ti.template()):
    lower, inc = self.get_inc()

    for i in range(self.size[0]):
      for j in range(self.size[1]):
        for k in range(self.size[2]):
          b = lower + vec3(i,j,k) * inc 
          box = AABox(b, b + inc)

          for l in range(objects.shape[0]):
            if objects[l].intersects_box(box):
              self.intersection[i,j,k].append(l)

  @ti.kernel
  def _get_counts(self) -> ti.math.ivec2:
    entries = 0
    n = 0
    
    for i in ti.grouped(self.box):
      entries +=  self.intersection[i.x, i.y, i.z].length()
      n += 1

    return ti.math.ivec2(n, entries)

  def get_counts(self):
    return self._get_counts()

    

  @ti.kernel
  def _subdivide(self, intersections:ti.template(), objects:ti.template()):
    lower, inc = self.get_inc()
    
    children = [
      vec3(0,0,0), vec3(0,0,1), vec3(0,1,0), vec3(0,1,1), 
      vec3(1,0,0), vec3(1,0,1), vec3(1,1,0), vec3(1,1,1)]

    for i in ti.grouped(intersections):
      idx = intersections[i]
      base = lower + vec3(i.x, i.y, i.z) * inc

      for offset in ti.static(children):
        b = base + offset * inc / 2.0
        box = AABox(b, b + inc / 2.0)
        
        if objects[idx].intersects_box(box):
          self.intersection[i.x, i.y, i.z].append(idx) 

  def subdivided(self, objects:ti.Field):
    grid = Grid(self.bounds, tuple(x * 2 for x in self.size))

    grid._subdivide(self.intersection, objects)
    return grid





if __name__ == "__main__":
  ti.init(arch=ti.gpu)

  torch_geom.AABox(lower = torch_geom.tensor([0,0,0]), 
    upper = torch_geom.tensor([1,1,1]))

  
