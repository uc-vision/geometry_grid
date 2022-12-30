import dataclasses
from typing import Tuple
import taichi as ti
from taichi.math import vec3, ivec3
import numpy as np

from geometry import torch as torch_geom
from geometry.torch.dataclass import TensorClass
from typeguard import typechecked

import torch
from geometry.taichi.types import AABox, Segment, Tube
from .conversion import from_torch, torch_field




@ti.func
def point_bounds(points:ti.template()) -> AABox:
  b = AABox(points[0], points[0])

  for i in points:
    b.lower = ti.math.min(b.lower, points[i])
    b.upper = ti.math.max(b.upper, points[i])
  return b






@ti.data_oriented
class Grid:
  def __init__(self, bounds:AABox, size:Tuple[int, int, int]):

    self.bounds = bounds 
    self.size = size


  def from_torch(box:torch_geom.AABox, 
    size:Tuple[int, int, int] | int): 
    assert box.shape == ()

    return Grid(
      bounds=torch_field(box.unsqueeze(0)),
      size = size if isinstance(size, tuple) else (size, size, size))
    

  @ti.func
  def get_inc(self) -> Tuple[vec3, vec3]:
    b = self.bounds[0]
    inc = (b.upper - b.lower) / self.size
    return b.lower, inc


  def add_objects(self, objects):
    
    self._intersect_dense(objects)
    self.objects = objects

  @ti.func 
  def grid_bounds(self, b):  
    lower, inc = self.get_inc()

    start = ti.floor((b.lower - lower) / inc)
    end = ti.ceil((b.upper - lower) / inc)

    return (ti.cast(ti.math.max(start, int(0)), ti.i32), 
            ti.cast(ti.math.min(end, ivec3(self.size)), ti.i32))


@ti.dataclass
class PointQuery:
    point: vec3
    radius: ti.f32

    distance: ti.f32
    index: ti.i32

    @ti.func
    def update(self, index, other):
      d = other.point_distance(self.point)
      old = ti.atomic_min(self.distance, d)
      if old == self.distance:
        self.index = index

    @ti.func
    def bounds(self) -> AABox:
      return AABox(self.point - self.radius, self.point + self.radius)

def point_query(points:torch.Tensor, radius:float):
    n = points.shape[0]
    queries = PointQuery.field(shape=n)
    queries.from_torch(dict(
      points = points, 
      radius = torch.full((n,), radius, dtype=torch.float32, device=points.device),
      distance = torch.full((n,), torch.inf, dtype=torch.float32, device=points.device),
      index = torch.full((n,), -1, dtype=torch.int32, device=points.device)
    ))

    return queries


@ti.data_oriented
class ObjectGrid:
  def __init__(self, grid:Grid, objects:ti.Field, max_occupied=64):

    self.occupied = ti.field(ti.i32)

    ptr:ti.SNode = ti.root.pointer(ti.ijk, [x//2 for x in grid.size])
    self.cells:ti.SNode = ptr.bitmasked(ti.ijk, (2,2,2))
    
    hits = self.cells.dynamic(ti.l, max_occupied, chunk_size=4)
    hits.place(self.occupied)

    self.objects = objects
    self._add_objects(objects)


  def from_torch(box:torch_geom.AABox, 
    size:Tuple[int, int, int] | int,
    objects:TensorClass,
    max_occupied=64): 

    assert box.shape == ()

    grid = Grid(
      bounds=torch_field(box.unsqueeze(0)),
      size = size if isinstance(size, tuple) else (size, size, size))

    return ObjectGrid(grid, from_torch(objects), max_occupied=max_occupied)

  @ti.kernel
  def _add_objects(self, objects:ti.template()):
    lower, inc = self.grid.get_inc()
    
    for l in range(objects.shape[0]):
      obj = objects[l]
      start, end = self.grid_bounds(obj.bounds())

      for i in range(start.x, end.x):
        for j in range(start.y, end.y):
          for k in range(start.z, end.z):
            box = AABox(lower + vec3(i,j,k) * inc, lower + vec3(i+1,j+1,k+1) * inc)
            if obj.intersects_box(box):
              self.occupied[i,j,k].append(l)


  @ti.kernel
  def _query(self, queries:ti.template()):

    for l in range(queries.shape[0]):
      query = queries[l]
      start, end = self.grid_bounds(query.bounds())
      for i in range(start.x, end.x):
        for j in range(start.y, end.y):
          for k in range(start.z, end.z):

            for idx in self.occupied[i,j,k]:
              query.update(idx, self.objects[idx])


  def point_query(self, points:torch.Tensor, radius:float):
    query = point_query(points, radius)
    self._query(query)

    result = query.to_torch()
    return result.index, result.distance


  @ti.kernel
  def _get_counts(self) -> ti.math.ivec2:
    entries = 0
    n = 0
    for i in ti.grouped(self.cells):
      entries +=  self.occupied[i.x, i.y, i.z].length()
      n += 1
    return ti.math.ivec2(n, entries)

  def get_counts(self):
    return tuple(int(x) for x in self._get_counts())


  @ti.kernel
  def _get_boxes(self, boxes:ti.template()):
    lower, inc = self.get_inc()
    for i in ti.grouped(self.cells):
      b = lower + vec3(i.x,i.y,i.z) * inc 
      box = AABox(b, b + inc)
      boxes.append(box)


  def get_boxes(self):
    n, _ = self.get_counts()
    boxes = AABox.field()
    ti.root.dynamic(ti.i, n).place(boxes)
    self._get_boxes(boxes)
    return boxes

  

if __name__ == "__main__":
  ti.init(arch=ti.gpu)

  torch_geom.AABox(lower = torch_geom.tensor([0,0,0]), 
    upper = torch_geom.tensor([1,1,1]))

  
