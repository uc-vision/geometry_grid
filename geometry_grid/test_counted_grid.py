

from typing import Tuple
from geometry_grid import taichi_geometry
from geometry_grid.render_util import display_distances
from geometry_grid.taichi_geometry.geometry_types import AABox
from geometry_grid.torch_geometry.random import random_segments

import taichi as ti
import torch
from geometry_grid.taichi_geometry.point_query import point_query

from geometry_grid.taichi_geometry.counted_grid import CountedGrid
from geometry_grid.taichi_geometry.transposed_query import build_grid_points_query
from geometry_grid.taichi_geometry.grid import Grid, morton_sort

import geometry_grid.torch_geometry as torch_geom
from tqdm import tqdm





if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--debug", action="store_true")
  parser.add_argument("--log", default=ti.INFO, choices=ti._logging.supported_log_levels)

  parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
  args = parser.parse_args()
 

  ti.init(arch=ti.cuda if args.device == "cuda" else ti.cpu, 
    debug=args.debug, 
    offline_cache=True,
    log_level=args.log)

  device = torch.device(args.device)
  torch.manual_seed(0)  

  segs = torch_geom.random_segments(torch_geom.AABox.from_to(0, 10, device=device), 
                                    length_range=(0.5, 3.0), n=1000).to(device)
  tubes = torch_geom.random_tubes(segs, radius_range=(0.1, 0.2))


  point_std = 0.05
  points = torch_geom.around_tubes(tubes, n=1000000, point_std=point_std)
  points = morton_sort(points, n=256)

  bounds = torch_geom.AABox.from_points(points)

  print("Generate grid...")
  grid = CountedGrid.from_torch(
    # Grid.fixed_cell(bounds,  1.0), 
    Grid.fixed_size(bounds, (64, 64, 64)), 
    torch_geom.Point(points))

  print("Grid size: ", grid.grid.size)
  cells, counts = grid.active_cells()
  
  query_points = build_grid_points_query(grid, taichi_geometry.Tube)

  print("Query index...")
  pbar = tqdm(range(10))
  for i in pbar:
    dist, idx = query_points(tubes, 0.5)
    pbar.set_description(f"n={(idx >= 0.0).sum().item()}")


  display_distances(tubes, grid.grid.get_boxes(cells), 
                    points, dist / (point_std * 3))


