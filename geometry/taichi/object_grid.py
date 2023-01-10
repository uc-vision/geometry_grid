from typing import Tuple
import taichi as ti
from taichi.math import vec3, ivec3, ivec2

from geometry.taichi.grid import Grid
from geometry.torch.dataclass import TensorClass
from typeguard import typechecked

import torch
from .conversion import from_torch
from taichi.types import ndarray




def block_bitmask(size, chunk):
    cell_blocks:ti.SNode = ti.root.bitmasked(ti.ijk, [1 + x//chunk for x in size])
    return cell_blocks.bitmasked(ti.ijk, (chunk,chunk,chunk))

@ti.data_oriented
class GridIndex:
  def __init__(self, grid:Grid, objects:ti.Field, cell_index:ti.Field, 
    index:ti.ndarray):
    
    self.grid = grid

    # Dense field of all objects
    self.objects = objects 

    # Sparse field (ivec2) which contains the pair
    # (index, count) which finds the entries in self.index
    self.cell_index = cell_index

    # List of all object indexes in cells, ordered by cell
    self.index = index 

  @ti.func
  def _query_cell(self, cell:ivec3, query:ti.template()):
    v = self.cell_index[cell] 
    index, count = v.x, v.y

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
    self.occupied = ti.field(ti.i32)
    self.grid_chunk=grid_chunk
    
    self.cells = block_bitmask(grid.size, grid_chunk)
    lists = self.cells.dynamic(ti.l, max_occupied, chunk_size=8)
    lists.place(self.occupied)

    self.grid = grid
    self.objects = objects
    self.total_cells, self.total_entries = [
      int(n) for n in self._add_objects(objects)]
    
    self.device = device

  @typechecked
  def from_torch(grid:Grid, objects:TensorClass, max_occupied=64): 
    return DynamicGrid(grid, from_torch(objects), 
      max_occupied=max_occupied, device=objects.device)

  @ti.func
  def _query_grid(self, query:ti.template()):
    ranges = self.grid.grid_ranges(query.bounds())

    for cell in ti.grouped(ti.ndrange(*ranges)):
        self._query_cell(cell, query)


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


  @ti.func
  def _query_cell(self, cell:ivec3, query:ti.template()):
    
    for l in range(self.occupied[cell.x, cell.y, cell.z].length()):
      idx = self.occupied[cell.x, cell.y, cell.z, l]
      query.update(idx, self.objects[idx])


  @ti.kernel
  def _active_cells(self, cells:ti.template(), 
    counts:ti.types.ndarray(ti.i32, ndim=1)):

    for i in ti.grouped(self.cells):
      cells.append(i)

    for i in range(self.total_cells):
      v = cells[i]
      counts[i] = self.occupied[v.x, v.y, v.z].length()



  def active_cells(self) -> Tuple[torch.Tensor, torch.Tensor]:
    cells = ivec3.field()
    l = ti.root.dynamic(ti.i, self.total_cells, 
      chunk_size=self.total_cells).place(cells)

    counts = torch.zeros(self.total_cells, device=self.device, dtype=torch.int32)

    self._active_cells(cells, counts)
    return cells.to_torch(), counts

  @ti.kernel
  def _fill_index(self, prefix:ndarray(ti.i32, ndim=1), 
    counts:ndarray(ti.i32, ndim=1), coords:ndarray(ivec3, ndim=1),
    cell_index:ti.template(), index:ti.template()):

    for i in range(self.total_cells):
      v = coords[i]
      p = ti.select(i > 0, prefix[i - 1], 0)

      for j in range(counts[i]):
        idx = self.occupied[v.x, v.y, v.z, j]
        index[p + j] = idx

      cell_index[v.x, v.y, v.z] = ivec2(p, counts[i])


  def make_index(self):

    coords, counts = self.active_cells()
    prefix = torch.cumsum(counts, dim=0, dtype=torch.int32)

    index = ti.field(ti.i32, self.total_entries)

    cell_index = ivec2.field()
    sparse = block_bitmask(self.grid.size, self.grid_chunk)
    sparse.place(cell_index)

    self._fill_index(prefix, counts, coords, cell_index, index)
    return GridIndex(self.grid, self.objects, cell_index, index)






@ti.data_oriented
class CountedGrid:
  def __init__(self, grid:Grid, objects:ti.Field, grid_chunk=8, device='cuda:0'):
    
    self.counts = ti.field(ti.i32)
    self.grid_chunk=grid_chunk
    
    self.cells = block_bitmask(grid.size, grid_chunk)
    self.counts.place(self.occupied)

    self.grid = grid
    self.objects = objects

    self.total_cells, self.total_entries = [
      int(n) for n in self._count_objects(objects)]
    
    self.device = device

  @typechecked
  def from_torch(grid:Grid, objects:TensorClass, grid_chunk=8): 
    return CountedGrid(grid, from_torch(objects), grid_chunk,  device=objects.device)

  # @ti.func
  # def _query_grid(self, query:ti.template()):
  #   ranges = self.grid.grid_ranges(query.bounds())

  #   for cell in ti.grouped(ti.ndrange(*ranges)):
  #       self._query_cell(cell, query)


  @ti.kernel
  def _count_objects(self, objects:ti.template()) -> ti.math.ivec2:
    total_entries = 0
    for l in range(objects.shape[0]):
      obj = objects[l]
      ranges = self.grid.grid_ranges(obj.bounds())
      
      for cell in ti.grouped(ti.ndrange(*ranges)):
        box = self.grid.cell_bounds(cell)
        if obj.intersects_box(box):
          total_entries += 1
          self.counts[cell.x, cell.y, cell.z] += 1

    total_cells = 0
    
    for _ in ti.grouped(self.cells):
      total_cells += 1

    return ti.math.vec2(total_cells, total_entries)


  @ti.func
  def _query_cell(self, cell:ivec3, query:ti.template()):
    
    for l in range(self.occupied[cell.x, cell.y, cell.z].length()):
      idx = self.occupied[cell.x, cell.y, cell.z, l]
      query.update(idx, self.objects[idx])


  @ti.kernel
  def _active_cells(self, cells:ti.template(), 
    counts:ti.types.ndarray(ti.i32, ndim=1)):

    for i in ti.grouped(self.cells):
      cells.append(i)

    for i in range(self.total_cells):
      v = cells[i]
      counts[i] = self.occupied[v.x, v.y, v.z].length()



  def active_cells(self) -> Tuple[torch.Tensor, torch.Tensor]:
    cells = ivec3.field()
    l = ti.root.dynamic(ti.i, self.total_cells, 
      chunk_size=self.total_cells).place(cells)

    counts = torch.zeros(self.total_cells, device=self.device, dtype=torch.int32)

    self._active_cells(cells, counts)
    return cells.to_torch(), counts

  @ti.kernel
  def _fill_index(self, prefix:ndarray(ti.i32, ndim=1), 
    counts:ndarray(ti.i32, ndim=1), coords:ndarray(ivec3, ndim=1),
    cell_index:ti.template(), index:ti.template()):

    for i in range(self.total_cells):
      v = coords[i]
      p = ti.select(i > 0, prefix[i - 1], 0)

      for j in range(counts[i]):
        idx = self.occupied[v.x, v.y, v.z, j]
        index[p + j] = idx

      cell_index[v.x, v.y, v.z] = ivec2(p, counts[i])


  def make_index(self):

    coords, counts = self.active_cells()
    prefix = torch.cumsum(counts, dim=0, dtype=torch.int32)

    index = ti.field(ti.i32, self.total_entries)

    cell_index = ivec2.field()
    sparse = block_bitmask(self.grid.size, self.grid_chunk)
    sparse.place(cell_index)

    self._fill_index(prefix, counts, coords, cell_index, index)
    return GridIndex(self.grid, self.objects, cell_index, index)



