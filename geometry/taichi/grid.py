import dataclasses
from typing import Tuple
import taichi as ti
from taichi.math import vec3, ivec3, clamp
from taichi.types import ndarray

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


  def fixed_size(box:torch_geom.AABox, size:Tuple[int, int, int] | int): 
    return Grid(bounds=from_aabox(box), size = ivec3(size))
    
  def fixed_cell(box:torch_geom.AABox, cell_size:float):
    grid_size = torch.ceil((box.upper - box.lower) / cell_size)
    bounds = torch_geom.AABox(box.lower, box.lower + grid_size * cell_size)

    return Grid(bounds=from_aabox(bounds), size = ivec3(grid_size.cpu().numpy()))


  @ti.func
  def get_inc(self) -> Tuple[vec3, vec3]:
    inc = (self.bounds.upper - self.bounds.lower) / self.size
    return self.bounds.lower, inc


  @ti.func 
  def grid_bounds(self, b):  
    lower, inc = self.get_inc()

    start = ti.floor((b.lower - lower) / inc)
    end = ti.ceil((b.upper - lower) / inc)

    return (ti.cast(clamp(start, 0, self.size), ti.i32), 
            ti.cast(clamp(end, 0, self.size), ti.i32))

  @ti.func 
  def grid_ranges(self, b):  
    lower, upper = self.grid_bounds(b)

    return ((lower.x, upper.x), (lower.y, upper.y), (lower.z, upper.z))


  @ti.func 
  def grid_cell(self, p:vec3) -> ivec3:
    lower, inc = self.get_inc()
    return ti.cast((p - lower) / inc, ti.i32)


  @ti.func
  def cell_bounds(self, cell:ivec3) -> AABox:
    lower, inc = self.get_inc()
    return AABox(lower + cell * inc, lower + (cell + 1) * inc)

  @ti.func
  def in_bounds(self, cell:ivec3) -> bool:
    return (0 <= cell).all() and (cell < self.size).all()

  @ti.kernel
  def _get_boxes(self, cells:ndarray(ivec3, ndim=1),
    lower:ndarray(vec3, ndim=1), 
    upper:ndarray(vec3, ndim=1)):
    for i in range(cells.shape[0]):
      v = cells[i]
      
      box = self.cell_bounds(ivec3(v.x, v.y, v.z))
      lower[i] = box.lower
      upper[i] = box.upper


  def get_boxes(self, cells:torch.Tensor):
    boxes = torch_geom.AABox.empty(cells.shape[0])
    self._get_boxes(cells, boxes.lower, boxes.upper)
    return boxes



if __name__ == "__main__":
  ti.init(arch=ti.gpu)

  torch_geom.AABox(lower = torch_geom.tensor([0,0,0]), 
    upper = torch_geom.tensor([1,1,1]))

  
