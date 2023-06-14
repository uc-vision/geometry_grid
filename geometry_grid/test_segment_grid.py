

from pathlib import Path
import gc
from typing import Tuple
from geometry_grid.taichi_geometry.geometry_types import AABox
from geometry_grid.torch_geometry.random import random_segments

import taichi as ti
import torch
from geometry_grid.taichi_geometry.point_query import point_query

from geometry_grid.taichi_geometry.grid import Grid
from geometry_grid.taichi_geometry.dynamic_grid import DynamicGrid

import geometry_grid.torch_geometry as torch_geom
from tqdm import tqdm


from open3d_vis import render
import open3d as o3d


def display_distances(segs:torch_geom.Segment, boxes, points, dist):
  red = torch.tensor([1.0, 0.0, 0.0], device=device)
  green = torch.tensor([0.0, 1.0, 0.0], device=device)

  t = (dist / 20.0).unsqueeze(1).clamp(0.0, 1.0)

  colors = (red * t + green * (1.0 - t)).squeeze()
  skel = render.segments(segs.a, segs.b, color=(0, 0, 1))

  o3d.visualization.draw([skel, boxes.render(), 
    render.point_cloud(points, colors=colors)])


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

  bounds = torch_geom.AABox.from_to(0, 10, device=device)
  segs = random_segments(bounds, length_range=(0.5, 3), n=100).to(device)
  
  points = torch_geom.around_segments(segs, n=1000000, radius=0.2)

  print("Generate grid...")
  grid = DynamicGrid.from_torch(
    Grid.fixed_cell(bounds,  1.0), 
    # Grid.fixed_size(skeleton.bounds, (64, 64, 64)), 
    segs,  max_occupied=64)


  print("Grid size: ", grid.grid.size)
  cells, counts = grid.active_cells()


  
  print("Query index...")
  pbar = tqdm(range(10))
  for i in pbar:
    dist, idx = point_query(grid.index, points, 20.0)
    pbar.set_description(f"n={(idx >= 0.0).sum().item()}")



  display_distances(segs, grid.grid.get_boxes(cells), points, dist)


