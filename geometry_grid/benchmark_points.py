import math

import taichi as ti
from taichi.math import vec3
import torch
from geometry_grid.taichi.dynamic_grid import DynamicGrid

from geometry_grid.taichi.grid import Grid, morton_sort
from geometry_grid.taichi.point_query import point_query


import geometry_grid.torch as torch_geom
from tqdm import tqdm


from open3d_vis import render
import open3d as o3d

from geometry_grid.torch.random import around_segments, random_segments


  
def display_densities(geom, boxes, counts, points):
  red = torch.tensor([1.0, 0.0, 0.0], device=points.device)
  green = torch.tensor([0.0, 1.0, 0.0], device=points.device)
  max_counts = counts.float().mean().item()

  t = (counts / max_counts).unsqueeze(1).clamp(0.0, 1.0)
  colors = (red * t + green * (1.0 - t)).squeeze()

  o3d.visualization.draw([*geom, boxes.render(colors), 
    render.point_cloud(points, colors=colors)])


def bench_query(name, grid_index, query_points, radius=0.2, n=100):
  print(f"Query {name}...")
  pbar = tqdm(range(n))
  for i in pbar:
    p = query_points + torch.randn_like(query_points) * i * (radius / n)

    dist, idx = point_query(grid_index, torch_geom.Point(p), radius)
    pbar.set_description(f"n={(idx >= 0.0).sum().item()} query_radius={radius} noise={i * 0.1}")


def bench_update_and_query(name, grid, points, radius):
  print(f"Update {name}...")
  pbar = tqdm(range(100))
  for i in pbar:
    p = points + torch.randn_like(points) * i * 0.5
    grid.update_objects(torch_geom.Point(p))

    dist, idx = point_query(grid.index, points, radius)
    pbar.set_description(f"n={(idx >= 0.0).sum().item()}  query_radius={radius} noise={(i * 0.1):.1f}")


def main(args):

  ti.init(arch=ti.cuda if args.device == "cuda" else ti.cpu, 
    debug=args.debug, 
    offline_cache=True,
    log_level=args.log,
    device_memory_fraction=0.75)

  device = torch.device(args.device)

  torch.manual_seed(0)  

  segs = random_segments(torch_geom.AABox.from_to(-5, 5, device=device), n=int(math.sqrt(args.n)))
  points = around_segments(segs, args.n, radius=0.2)

  points = morton_sort(points, n=256)
  query_points = points + torch.randn_like(points) * 0.1

  point_objs = torch_geom.Point(points)
  bounds = point_objs.bounds.merge()

  grid = Grid.fixed_size(bounds, (64, 64, 64))
  
  # print("Generate grid...")
  # point_grid = CountedGrid.from_torch(grid, point_objs, grid_chunk=8)
 

  print("Generate grid...")
  dyn_grid = DynamicGrid.from_torch(grid, point_objs, 
    grid_chunk=8, max_occupied=256)
  

  bench_update_and_query("Dynamic grid", dyn_grid, points, 0.2)


  bench_query("Dynamic grid", dyn_grid.index, query_points)
  # bench_query("Counted grid", point_grid, query_points)

 

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--debug", action="store_true")
  parser.add_argument("--n", type=int, default=100000)
  parser.add_argument("--log", default=ti.INFO, choices=ti._logging.supported_log_levels)
  parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

  args = parser.parse_args()
 
  main(args)




  


