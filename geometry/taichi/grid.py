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


def from_aabox(box:torch_geom.AABox):
  assert box.shape == ()
  l, u = [x.cpu().numpy() for x in [box.lower, box.upper]]
  return AABox(l, u)


@ti.data_oriented
class Grid:
  @typechecked
  def __init__(self, bounds:AABox, size:ivec3):

    self.bounds = bounds 
    self.size = size


  def from_torch(box:torch_geom.AABox, size:Tuple[int, int, int] | int): 
    return Grid(bounds=from_aabox(box), size = ivec3(size))
    

  @ti.func
  def get_inc(self) -> Tuple[vec3, vec3]:
    inc = (self.bounds.upper - self.bounds.lower) / self.size
    return self.bounds.lower, inc


  @ti.func 
  def grid_bounds(self, b):  
    lower, inc = self.get_inc()

    start = ti.floor((b.lower - lower) / inc)
    end = ti.ceil((b.upper - lower) / inc)

    return (ti.cast(ti.math.max(start, int(0)), ti.i32), 
            ti.cast(ti.math.min(end, ivec3(self.size)), ti.i32))

  @ti.func
  def cell_bounds(self, cell:ivec3) -> AABox:
    lower, inc = self.get_inc()
    return AABox(lower + cell * inc, lower + (cell + 1) * inc)



@ti.dataclass
class PointQuery:
    point: vec3
    radius: ti.f32

    distance: ti.f32
    index: ti.i32
    comparisons: ti.i32

    @ti.func
    def update(self, index, other):
      d = other.point_distance(self.point)
      if d < self.radius:
        old = ti.atomic_min(self.distance, d)
        if old == self.distance:
          self.index = index


    @ti.func
    def bounds(self):
      lower = self.point - self.radius
      upper = self.point + self.radius
      return AABox(lower, upper)





@ti.data_oriented
class ObjectGrid:
  def __init__(self, grid:Grid, objects:ti.Field, max_occupied=64, grid_chunk=4, device='cuda:0'):

    self.occupied = ti.field(ti.i32)
    level1:ti.SNode = ti.root.bitmasked(ti.ijk, [x//grid_chunk for x in grid.size])
    self.cells:ti.SNode = level1.bitmasked(ti.ijk, (grid_chunk,grid_chunk,grid_chunk))
    lists = self.cells.dynamic(ti.l, max_occupied, chunk_size=8)
    lists.place(self.occupied)

    self.grid = grid
    self.objects = objects
    self.total_cells, self.total_entries = [
      int(n) for n in self._add_objects(objects)]
    
    self.device = device

  @typechecked
  def from_torch(box:torch_geom.AABox, 
    size:Tuple[int, int, int] | int,
    objects:TensorClass,
    max_occupied=64): 

    grid =  Grid(bounds=from_aabox(box), size = ivec3(size))
    return ObjectGrid(grid, from_torch(objects), max_occupied=max_occupied)

  @ti.kernel
  def _add_objects(self, objects:ti.template()) -> ti.math.ivec2:
    total_entries = 0
    for l in range(objects.shape[0]):
      obj = objects[l]
      start, end = self.grid.grid_bounds(obj.bounds())

      for i in range(start.x, end.x):
        for j in range(start.y, end.y):
          for k in range(start.z, end.z):
            box = self.grid.cell_bounds(ivec3(i,j,k))

            if obj.intersects_box(box):
              total_entries += 1
              self.occupied[i,j,k].append(l)

    total_cells = 0
    
    for i,j,k in self.cells:
      total_cells += 1

    return ti.math.vec2(total_cells, total_entries)

  
  @ti.func
  def grid_bounds(self, obj):
    return self.grid.grid_bounds(obj)

  @ti.func
  def _run_query(self, query:ti.template()):
      start, end = self.grid_bounds(query.bounds())

      # ti.loop_config(serialize=True)
      for i in range(start.x, end.x):

        # ti.loop_config(serialize=True)
        for j in range(start.y, end.y):

          # ti.loop_config(serialize=True)
          for k in range(start.z, end.z):

            # ti.loop_config(serialize=True)
            for l in range(self.occupied[i,j,k].length()):
              idx = self.occupied[i,j,k,l]
              query.update(idx, self.objects[idx])

  @ti.kernel
  def _point_query(self, 
    points:ti.types.ndarray(vec3, ndim=1), radius:ti.f32,
    distances:ti.types.ndarray(ti.f32, ndim=1),
    indexes:ti.types.ndarray(ti.i32, ndim=1)):
    
    for i in range(points.shape[0]):
      q = PointQuery(points[i], radius, distance=torch.inf, index=-1)
      self._run_query(q)

      distances[i] = q.distance
      indexes[i] = q.index


  def point_query(self, points:torch.Tensor, radius:float) -> Tuple[torch.FloatTensor, torch.IntTensor]:
    distances = torch.empty((points.shape[0],), device=points.device, dtype=torch.float32)
    indexes = torch.empty_like(distances, dtype=torch.int32)

    self._point_query(points, radius, distances, indexes)
    return distances, indexes


  @ti.kernel
  def _active_cells(self, cells:ti.template(), 
    counts:ti.types.ndarray(ti.i32, ndim=1),
    lower:ti.types.ndarray(vec3, ndim=1),
    upper:ti.types.ndarray(vec3, ndim=1)
    ):

    for i in ti.grouped(self.cells):
      cells.append(ivec3(i.x, i.y, i.z))

    for i in range(self.total_cells):
      v = cells[i]
      counts[i] = self.occupied[v.x, v.y, v.z].length()

      box = self.grid.cell_bounds(ivec3(v.x, v.y, v.z))
      lower[i] = box.lower
      upper[i] = box.upper


  def active_cells(self):
    cells = ivec3.field()
    l = ti.root.dynamic(ti.i, self.total_cells, 
      chunk_size=self.total_cells).place(cells)

    counts = torch.zeros(self.total_cells, device=self.device)
    boxes = torch_geom.AABox.empty(self.total_cells)

    self._active_cells(cells, counts, boxes.lower, boxes.upper)
    return cells.to_torch(), counts, boxes





if __name__ == "__main__":
  ti.init(arch=ti.gpu)

  torch_geom.AABox(lower = torch_geom.tensor([0,0,0]), 
    upper = torch_geom.tensor([1,1,1]))

  
