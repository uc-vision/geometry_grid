

from typing import Tuple
import torch

import taichi as ti

from geometry_grid.taichi.dynamic_grid import DynamicGrid
from geometry_grid.taichi.point_distances import min_distances, distances
from geometry_grid.taichi.point_query import point_query

import geometry_grid.torch as torch_geom
import geometry_grid.taichi as ti_geom

from geometry_grid.torch.random import around_segments, random_segments
from geometry_grid.torch.typecheck import typechecked



@typechecked
def run_test(bounds:torch_geom.AABox, 
  grid_size:Tuple[int, int, int], 
  n:int = 1000, 
  seg_length:float = 1.0,

  n_points:int = 100000, 
  radius:float=0.5):

  segs = random_segments(bounds, length_range=(seg_length * 0.5, seg_length * 2.0), n=n)
  
  points = around_segments(segs, n_points, radius)

  grid = ti_geom.Grid.fixed_size(bounds, grid_size)
  obj_grid = DynamicGrid.from_torch(grid, ti_geom.Segment, segs, max_occupied=64)

  dist1, idx1 = point_query(obj_grid.index, points, 1.0)
  dist2, idx2 = min_distances(segs, points, 1.0)

  assert torch.sum(idx1 >= 0) == torch.sum(idx2 >= 0)
  assert torch.allclose(dist1, dist2)

  valid = idx1[idx1 >= 0].long()
  dist3 = distances(segs[valid], points[idx1 >= 0])

  assert torch.allclose(dist1[idx1 >= 0], dist3)


def test_grid(i):
  torch.manual_seed(i)
  grid_size = tuple(torch.randint(4, 32, (3,)).tolist())

  n = int(torch.randint(10, 100, ()).item())
  seg_length = (torch.rand(()) * 5.0 + 0.1).item()

  n_points = int(torch.randint(100, 1000, ()).item())
  radius = (torch.rand(()) * 5.0 + 0.1).item()

  bounds = torch_geom.AABox(torch.tensor([-10.0, -10.0, -10.0]), 
    torch.tensor([10.0, 10.0, 10.0]))

  print(f"{i}: grid_size={grid_size}, n={n}, seg_length={seg_length}, n_points={n_points}, radius={radius}")

  run_test(bounds, grid_size, n, seg_length, n_points, radius)


if __name__ == "__main__":
  ti.init(arch=ti.cpu, debug=True, offline_cache=True,
    log_level=ti.DEBUG)

  for i in range(5, 45):
    test_grid(i)