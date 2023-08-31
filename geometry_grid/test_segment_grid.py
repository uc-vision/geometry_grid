from geometry_grid.render_util import display_distances

import taichi as ti
import torch
from geometry_grid.taichi_geometry.point_query import point_query

from geometry_grid.taichi_geometry.grid import Grid, morton_sort
from geometry_grid.taichi_geometry.dynamic_grid import DynamicGrid

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

  segs = torch_geom.random_segments(torch_geom.AABox.from_to(0, 50, device=device), 
                                    length_range=(0.5, 5.0), n=10000).to(device)
  tubes = torch_geom.random_tubes(segs, radius_range=(0.1, 0.4))

  bounds = tubes.bounds.union_all()

  point_std = 0.1
  points = torch_geom.around_tubes(tubes, n=1000000, point_std=point_std)
  points = morton_sort(points, n=256)

  print("Generate grid...")
  dyn_grid = DynamicGrid.from_torch(
    # Grid.fixed_cell(bounds,  1.0), 
    Grid.fixed_size(bounds, (16, 16, 16)), 
    tubes,  max_occupied=64)

  print("Grid size: ", dyn_grid.grid.size)
  cells, counts = dyn_grid.active_cells()
  
  print("Query index...")
  pbar = tqdm(range(10))
  for i in pbar:
    dist, idx = point_query(dyn_grid.index, points, 0.5)
    pbar.set_description(f"n={(idx >= 0.0).sum().item()}")



  display_distances(tubes, dyn_grid.grid.get_boxes(cells), 
                    points, dist / (point_std * 3))


