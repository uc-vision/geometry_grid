from typing import Tuple
import taichi as ti
from taichi.math import vec3, ivec3, ivec2
from geometry_grid.taichi.field import placed_field

from geometry_grid.taichi.grid import Grid
from geometry_grid.torch.dataclass import TensorClass
from typeguard import typechecked

import torch
from .conversion import from_torch
from taichi.types import ndarray

from .dynamic_grid import block_bitmask, GridIndex




@ti.data_oriented
class CountedGrid:
  def __init__(self, grid:Grid, objects:ti.Field, grid_chunk=8, device='cuda:0'):
    self.grid = grid
    self.objects = objects

    self.grid_chunk=grid_chunk
    self.grid_size_chunks = grid.size // self.grid_chunk
    
    self.cells = block_bitmask(grid.size, grid_chunk)
    self.chunks = self.cells.parent()

    self.counts = placed_field(self.cells, ti.i32)
    self.prefix = placed_field(self.cells, ti.i32)
    
    self.chunk_counts = placed_field(self.chunks, ti.i32)
    self.chunk_prefix = placed_field(self.chunks, ti.i32)
    
    self.current_index = placed_field(self.cells, ti.i32)

    self.total_cells, self.total_entries = [
      int(n) for n in self._count_objects()]

    self._compute_prefixes()

    self.index = ti.field(ti.i32, self.total_entries)
    self._fill_index()

    self.device = device


  @typechecked
  def from_torch(grid:Grid, objects:TensorClass, grid_chunk=8): 
    return CountedGrid(grid, from_torch(objects), 
      grid_chunk,  device=objects.device)

  @ti.kernel
  def _compute_prefixes(self):
    chunk_prefix = 0
    n = self.grid_size_chunks

    ti.loop_config(serialize=True)
    for i in ti.grouped(ti.ndrange(n.x, n.y, n.z)):
      if ti.is_active(self.chunks, i):
        self.chunk_prefix[i] = chunk_prefix 
        chunk_prefix += self.chunk_counts[i]

    for i in ti.grouped(self.chunks):
      prefix = self.chunk_prefix[i]

      ti.loop_config(serialize=True)
      for j in ti.grouped(ti.ndrange(self.grid_chunk, self.grid_chunk, self.grid_chunk)):
        cell = i * self.grid_chunk + j
        if ti.is_active(self.cells, cell):
          self.prefix[cell] = prefix          
          prefix += self.counts[cell]


  @ti.kernel
  def _fill_index(self):

    # Initialise index with the prefix sum
    for cell in ti.grouped(self.cells):
      self.current_index[cell] = self.prefix[cell]

    for l in range(self.objects.shape[0]):
      ranges = self.grid.grid_ranges(self.objects[l].bounds())

      for cell in ti.grouped(ti.ndrange(*ranges)):
        box = self.grid.cell_bounds(cell)
        if self.objects[l].intersects_box(box):
          # Increment pointer and add object to the index
          index_loc = ti.atomic_add(self.current_index[cell], 1)
          self.index[index_loc] = l

    

  @ti.kernel
  def _count_objects(self) -> ti.math.ivec2:
    total_entries = 0

    for l in range(self.objects.shape[0]):
      ranges = self.grid.grid_ranges(self.objects[l].bounds())
      for cell in ti.grouped(ti.ndrange(*ranges)):
        box = self.grid.cell_bounds(cell)
        if self.objects[l].intersects_box(box):
          total_entries += 1
          self.counts[cell.x, cell.y, cell.z] += 1

          chunk = cell // self.grid_chunk
          self.chunk_counts[chunk.x, chunk.y, chunk.z] += 1

    total_cells = 0
    
    for _ in ti.grouped(self.cells):
      total_cells += 1

    return ti.math.vec2(total_cells, total_entries)


  @ti.func
  def _query_cell(self, cell:ivec3, query:ti.template()):
    index, count = self.prefix[cell], self.counts[cell]

    for l in range(count):
      idx = self.index[index + l]
      query.update(idx, self.objects[idx])


  @ti.func
  def _query_grid(self, query:ti.template()):
    ranges = self.grid.grid_ranges(query.bounds())
    for i in ti.grouped(ti.ndrange(*ranges)):
        self._query_cell(i, query)



  @ti.kernel
  def _active_cells(self, cells:ti.template(), 
    counts:ti.types.ndarray(ti.i32, ndim=1)):

    for i in ti.grouped(self.cells):
      cells.append(i)

    for i in range(self.total_cells):
      counts[i] = self.counts[cells[i]]


  def active_cells(self) -> Tuple[torch.Tensor, torch.Tensor]:
    cells = ivec3.field()
    l = ti.root.dynamic(ti.i, self.total_cells, 
      chunk_size=self.total_cells).place(cells)

    counts = torch.zeros(self.total_cells, device=self.device, dtype=torch.int32)

    self._active_cells(cells, counts)
    return cells.to_torch(), counts



    



    