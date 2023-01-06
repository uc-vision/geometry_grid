

from pathlib import Path
import gc
from typing import Tuple

import taichi as ti
import torch
from geometry.torch.loading import display_skeleton, load_tree

from geometry.taichi.grid import Grid
from geometry.taichi.object_grid import DynamicGrid
from geometry.taichi.point_distances import min_distances, distances

import geometry.torch as torch_geom
from tqdm import tqdm


from open3d_vis import render
import open3d as o3d


def display_distances(skeleton, boxes, points, dist):
  red = torch.tensor([1.0, 0.0, 0.0], device=device)
  green = torch.tensor([0.0, 1.0, 0.0], device=device)

  t = (dist / 20.0).unsqueeze(1).clamp(0.0, 1.0)

  colors = (red * t + green * (1.0 - t)).squeeze()

  skel = render.line_set(skeleton.points, skeleton.edges)
  o3d.visualization.draw([skel, boxes.render(), 
    render.point_cloud(points, colors=colors)])


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("filename", type=Path)
  parser.add_argument("--debug", action="store_true")
  parser.add_argument("--log", default=ti.INFO, choices=ti._logging.supported_log_levels)

  parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

  args = parser.parse_args()
 

  ti.init(arch=ti.cuda if args.device == "cuda" else ti.cpu, 
    debug=args.debug, 
    offline_cache=True,
    log_level=args.log)

  device = torch.device(args.device)

  skeleton = load_tree(args.filename, radius_threshold=0)
  print("Extents: ", skeleton.bounds.extents)

  segs = skeleton.tubes.segment.to(device)

  torch.manual_seed(0)  
  points = torch_geom.around_segments(segs, 20.0, 1000000)

  print("Generate grid...")
  grid = DynamicGrid.from_torch(
    Grid.fixed_cell(skeleton.bounds,  10.0), 
    # Grid.fixed_size(skeleton.bounds, (64, 64, 64)), 
    segs,  max_occupied=64)



  print("Grid size: ", grid.grid.size)
  cells, counts = grid.active_cells()

  print("Query grid...")
  pbar = tqdm(range(10))
  for i in pbar:
    dist, idx = grid.point_query(points, 20.0)
    pbar.set_description(f"n={(idx >= 0.0).sum().item()}")

  index = grid.make_index()
  
  print("Query index...")
  pbar = tqdm(range(10))
  for i in pbar:
    dist, idx = index.point_query(points, 20.0)
    pbar.set_description(f"n={(idx >= 0.0).sum().item()}")



  display_distances(skeleton, grid.grid.get_boxes(cells), points, dist)


