from dataclasses import asdict
from typing import Tuple
import taichi as ti
from taichi.math import vec3, ivec3, ivec2
from geometry.taichi.field import block_bitmask, placed_field

from geometry.taichi.grid import Grid
from geometry.torch.dataclass import TensorClass
from typeguard import typechecked

import torch
from .conversion import from_torch
from taichi.types import ndarray



@ti.data_oriented
class GridIndex:
  def __init__(self, grid:Grid, objects:ti.Field, total_entries:int, grid_chunk:int=8):
    self.grid = grid

    # Dense field of all objects
    self.objects = objects 

    # Sparse fields which contains the pair
    # (prefix, count) which points at ranges in self.index
    self.sparse = block_bitmask(grid.size, grid_chunk)
    self.prefix = placed_field(self.sparse, ti.int32)
    self.count = placed_field(self.sparse, ti.int32)

    # Flattened ragged lists of indexes into objects for each cell
    self.index = ti.field(ti.i32, total_entries)
    self.num_entries = total_entries
  

  def clear(self):
    self.sparse.deactivate_all()

  @ti.func
  def _query_cell(self, cell:ivec3, query:ti.template()):
    index = self.prefix[cell] 
    count = self.count[cell] 

    for l in range(count):
      idx = self.index[index + l]
      query.update(idx, self.objects[idx])


  @ti.func
  def _query_grid(self, query:ti.template()):
    ranges = self.grid.grid_ranges(query.bounds())

    for i in ti.grouped(ti.ndrange(*ranges)):
        self._query_cell(i, query)





@ti.data_oriented
class DynamicGrid:
  def __init__(self, grid:Grid, objects:ti.Field, max_occupied=64, 
    grid_chunk=8, device='cuda:0'):

    self.device = device
    
    self.occupied = ti.field(ti.i32)
    self.grid_chunk=grid_chunk
    
    self.cells = block_bitmask(grid.size, grid_chunk)
    lists = self.cells.dynamic(ti.l, max_occupied, chunk_size=4)
    lists.place(self.occupied)

    self.grid = grid
    self.objects = objects
    self.index = None

    self.add_objects()
    


  @typechecked
  def from_torch(grid:Grid, objects:TensorClass, grid_chunk=8, max_occupied=64): 
    return DynamicGrid(grid, from_torch(objects), 
      max_occupied=max_occupied, device=objects.device, grid_chunk=grid_chunk)


  def update_objects(self, objects:TensorClass):
    self.objects.from_torch(asdict(objects))
    self.cells.parent().deactivate_all()
    self.add_objects()


  def add_objects(self):
    self.total_cells, self.total_entries = [
      int(n) for n in self._add_objects(self.objects)]
    self.update_index()


  @ti.kernel
  def _add_objects(self, objects:ti.template()) -> ti.math.ivec2:
    total_entries = 0
    for l in range(objects.shape[0]):
      obj = objects[l]
      ranges = self.grid.grid_ranges(obj.bounds())
      
      for cell in ti.grouped(ti.ndrange(*ranges)):
        box = self.grid.cell_bounds(cell)
        if obj.intersects_box(box):
          total_entries += 1
          self.occupied[cell.x, cell.y, cell.z].append(l)

    total_cells = 0
    
    for _ in ti.grouped(self.cells):
      total_cells += 1
    return ti.math.vec2(total_cells, total_entries)


  @ti.kernel
  def _active_cells(self, cells:ti.types.ndarray(ivec3, ndim=1), 
    counts:ti.types.ndarray(ti.i32, ndim=1)):
    count = 0

    for cell in ti.grouped(self.cells):
      i = ti.atomic_add(count, 1)
      v = ivec3(cell)
      cells[i] = v
      counts[i] = self.occupied[v.x, v.y, v.z].length()



  def active_cells(self) -> Tuple[torch.Tensor, torch.Tensor]:
    cells = torch.zeros((self.total_cells, 3), device=self.device, dtype=torch.int32)
    counts = torch.zeros(self.total_cells, device=self.device, dtype=torch.int32)

    self._active_cells(cells, counts)
    return cells, counts

  @ti.kernel
  def _fill_index(self, index:ti.template(), prefix:ndarray(ti.i32, ndim=1), counts:ndarray(ti.i32, ndim=1), 
    coords:ndarray(ivec3, ndim=1), ):

    for i in range(self.total_cells):
      v = coords[i]
      p = ti.select(i > 0, prefix[i - 1], 0)

      for j in range(counts[i]):
        idx = self.occupied[v.x, v.y, v.z, j]
        index.index[p + j] = idx

      index.prefix[v] = p
      index.count[v] = counts[i]



  def update_index(self):
    if (self.index is None) or (self.total_entries > self.index.num_entries):
      self.index = GridIndex(self.grid, self.objects, 
        self.total_entries * 2, self.grid_chunk)
    else:
      self.index.clear()

    coords, counts = self.active_cells()
    prefix = torch.cumsum(counts, dim=0, dtype=torch.int32)

    self._fill_index(self.index, prefix, counts, coords)





