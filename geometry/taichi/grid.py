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

from taichi.algorithms import parallel_sort


def point_bounds(points:torch.Tensor) -> torch_geom.AABox:
  assert points.shape[1] == 3
  return torch_geom.AABox(points.min(dim=0).values, points.max(dim=0).values)



def from_aabox(box:torch_geom.AABox):
  assert box.shape == ()
  l, u = [x.cpu().numpy() for x in [box.lower, box.upper]]
  return AABox(l, u)

# https://stackoverflow.com/questions/
# 1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints 
@ti.func
def spreads_bits32(x:ti.int32) -> ti.int32:
  x &= 0x3ff
  x = (x | (x << 16)) & 0x030000FF
  x = (x | (x <<  8)) & 0x0300F00F
  x = (x | (x <<  4)) & 0x030C30C3
  x = (x | (x <<  2)) & 0x09249249
  return x

@ti.func
def spreads_bits64(x:ti.int64) -> ti.int64:
  x &= 0x1fffff
  x = (x | (x << 32)) & 0x1f00000000ffff
  x = (x | (x << 16)) & 0x1f0000ff0000ff
  x = (x | (x << 8)) & 0x100f00f00f00f00f
  x = (x | (x << 4)) & 0x10c30c30c30c30c3
  x = (x | (x << 2)) & 0x1249249249249249
  return x


@typechecked
def morton_sort(points:torch.Tensor, bounds=None, n=1024):
  if bounds is None:
    bounds = from_aabox(point_bounds(points))
    
  return Grid(bounds, ivec3(n)).morton_sort(points)

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
    v = (p - lower) / inc
    return ti.cast(clamp(v, 0, self.size - 1), ti.i32)


  @ti.func
  def cell_bounds(self, cell:ivec3) -> AABox:
    lower, inc = self.get_inc()
    return AABox(lower + cell * inc, lower + (cell + 1) * inc)

  @ti.func
  def in_bounds(self, cell:ivec3) -> bool:
    return (0 <= cell).all() and (cell < self.size).all()

  @ti.func
  def assert_in_bounds(self, v:ivec3):
    s = self.size
    assert  self.in_bounds(v), f"{v.x} {v.y} {v.z} not in {s.x} {s.y} {s.z}"


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

  @ti.func
  def cell_code64(self, cell:ivec3) -> ti.int64:
    cell = ti.cast(cell, ti.int64)
    return (spreads_bits64(cell.x) 
      | (spreads_bits64(cell.y) << 1) 
      | (spreads_bits64(cell.z) << 2))

  @ti.func
  def morton_code64(self, p:vec3) -> ti.int64:
    cell = self.grid_cell(p)
    return self.cell_code64(cell)


  @ti.func
  def cell_code32(self, cell:ivec3) -> ti.int32:
    return (spreads_bits32(cell.x) 
      | (spreads_bits32(cell.y) << 1) 
      | (spreads_bits32(cell.z) << 2))

  @ti.func
  def morton_code32(self, p:vec3) -> ti.int32:
    cell = self.grid_cell(p)
    return self.cell_code32(cell)

  @ti.kernel
  def _code_points(self, points:ndarray(vec3, ndim=1), 
    codes:ndarray(ti.int32, ndim=1)):
    for i in range(points.shape[0]):
      codes[i] = self.morton_code32(points[i])

  def morton_argsort(self, points:torch.Tensor):
    codes = torch.zeros(points.shape[0], dtype=torch.int32, device=points.device)
    self._code_points(points, codes)
    return torch.argsort(codes)

  def morton_sort(self, points:torch.Tensor):
    return points[self.morton_argsort(points)]
    


if __name__ == "__main__":
  ti.init(arch=ti.gpu)

  torch_geom.AABox(lower = torch_geom.tensor([0,0,0]), 
    upper = torch_geom.tensor([1,1,1]))

  
