

from pathlib import Path

import taichi as ti
from taichi.math import vec3
import torch
from geometry.taichi.types import AABox
from geometry.torch.loading import display_skeleton, load_tree

from geometry.taichi.grid import Grid
from geometry.taichi.counted_grid import CountedGrid

from geometry.taichi.point_query import point_query


import geometry.torch as torch_geom
from tqdm import tqdm


from open3d_vis import render
import open3d as o3d


  
def display_densities(geom, boxes, counts, points):
  red = torch.tensor([1.0, 0.0, 0.0], device=points.device)
  green = torch.tensor([0.0, 1.0, 0.0], device=points.device)
  max_counts = counts.float().mean().item()

  t = (counts / max_counts).unsqueeze(1).clamp(0.0, 1.0)
  colors = (red * t + green * (1.0 - t)).squeeze()

  o3d.visualization.draw([*geom, boxes.render(colors), 
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

  tubes = skeleton.tubes.to(device)

  torch.manual_seed(0)  
  points = torch_geom.around_tubes(tubes, 1000000)



  point_objs = torch_geom.Point(points)
  bounds = point_objs.bounds.merge()

  grid = Grid.fixed_size(bounds, (128, 128, 128))
  order = grid.morton_argsort(points)
  

  print("Generate grid...")
  point_grid = CountedGrid.from_torch(grid, point_objs[order])
 
  # print("Grid size: ", point_grid.grid.size)
  # cells, counts = point_grid.active_cells()


  # skel = display_skeleton(skeleton)
  # display_densities([skel], point_grid.grid.get_boxes(cells), counts, points)


  uniform_points = grid.morton_sort(
      bounds.random_points(10000000))


  print("Query grid...")
  pbar = tqdm(range(10))
  for i in pbar:
    dist, idx = point_query(point_grid, uniform_points, 5.0)
    pbar.set_description(f"n={(idx >= 0.0).sum().item()}")



  

