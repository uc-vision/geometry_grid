

from typing import Tuple
import torch

from geometry_grid.taichi_geometry.dynamic_grid import DynamicGrid

import geometry_grid.torch_geometry as torch_geom
import geometry_grid.taichi_geometry as ti_geom

from geometry_grid.torch_geometry.random import around_segments, random_segments
from geometry_grid.torch_geometry.typecheck import typechecked


@typechecked
def test_with(f,
  bounds:torch_geom.AABox, 
  grid_size:Tuple[int, int, int], 
  n:int = 1000, 
  seg_length:float = 1.0,

  radius:float=0.5,
  noise_std=0.2):

  segs = random_segments(bounds, 
      length_range=(seg_length * 0.5, seg_length * 2.0), n=n)


  grid = ti_geom.Grid.fixed_size(bounds, grid_size)
  obj_grid = DynamicGrid.from_torch(grid, segs, max_occupied=64)

  for i in range(1000):
    n_points = int(torch.randint(10, 100, ()).item())

    points = around_segments(segs, n_points, radius, point_std=noise_std)
    f(obj_grid, segs, points, radius)


def test_grid_with(f, seed=0):
  print(f"seed={seed}")
  torch.manual_seed(seed)
  grid_size = tuple(torch.randint(4, 32, (3,)).tolist())

  n = int(torch.randint(10, 100, ()).item())
  seg_length = (torch.rand(()) * 5.0 + 0.1).item()

  radius = (torch.rand(()) * 1.0 + 0.1).item()
  noise_std = torch.rand(() ).item() * radius


  bounds = torch_geom.AABox(torch.tensor([-10.0, -10.0, -10.0]), 
    torch.tensor([10.0, 10.0, 10.0]))

  print(f"grid_size={grid_size}, n={n}, seg_length={seg_length}, radius={radius} noise_std={noise_std}")
  test_with(f, bounds, grid_size, n, seg_length, radius)


