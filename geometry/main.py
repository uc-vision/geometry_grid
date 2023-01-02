

from pathlib import Path
import gc

import taichi as ti
import numpy as np
import torch
from geometry.taichi.conversion import from_torch, torch_field
from geometry.taichi.types import Segment
from geometry.torch.dataclass import TensorClass
from geometry.torch.loading import display_skeleton, load_tree

from geometry.taichi.grid import Grid, ObjectGrid
import geometry.taichi as ti_geom
import geometry.torch as torch_geom
from tqdm import tqdm


from open3d_vis import render
import open3d as o3d



@ti.func
def point_bounds(points:ti.template(dim=1)) -> ti_geom.AABox:
  bounds = ti_geom.AABox(points[0], points[0])
  for i in points:
    bounds = bounds.union(ti_geom.AABox(points[i], points[i]))
  return bounds


def around_segments(segments:torch_geom.Segment, radius:float, n:int):
  i = torch.randint(low=0, high=segments.shape[0] -1, 
    size=(n,), device=segments.device)

  t = torch.rand(n, device=segments.device)
  r = torch.rand(n, 3, device=segments.device) * 2.0  - 1.0
  return segments[i].points(t) + r * radius



@ti.kernel
def _pairwise_min_distance(objects:ti.template(), 
  points:ti.types.ndarray(ti.math.vec3),
  radius:ti.f32,
  distances:ti.types.ndarray(ti.f32),
  indices:ti.types.ndarray(ti.i32)):

  for j in range(points.shape[0]):
    ti.loop_config(serialize=True)
    min_d = torch.inf
    index = -1

    for i in range(objects.shape[0]):
      d = objects[i].point_distance(points[j])
      if d < min_d and d < radius:
        min_d = d
        index = i

    distances[j] = min_d
    indices[j] = index
        

def pairwise_min_distance(objects:TensorClass, points:torch.Tensor, radius:float):
  distances = torch.full((points.shape[0],), torch.inf, device=points.device, dtype=torch.float32)
  indexes = torch.full_like(distances, -1, dtype=torch.int32)

  objs = from_torch(objects)
  _pairwise_min_distance(objs, points, radius, distances, indexes)
  return distances, indexes


@ti.kernel
def point_bounds_(points:ti.types.ndarray(dtype=ti.math.vec3), bounds:ti.template()):
  bounds[0] = point_bounds(points)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("filename", type=Path)
  parser.add_argument("--debug", action="store_true")
  parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

  args = parser.parse_args()

  ti.init(arch=ti.cuda, debug=args.debug, offline_cache=True,
    log_level=ti.DEBUG)

  skeleton = load_tree(args.filename, radius_threshold=0)
  print("Extents: ", skeleton.bounds.extents)

  segs = skeleton.tubes.segment

  torch.set_deterministic_debug_mode(True)
  torch.manual_seed(0)
  points = around_segments(segs, 1, 100000)
  
  print("Generate grid...")
  grid = ObjectGrid.from_torch(skeleton.bounds, 64, segs, max_occupied=32)
  cells, counts, boxes = grid.active_cells()


  dist1, idx1 = grid.point_query(points, 10.0)
  print(dist1[idx1 > 0].max(), torch.sum(idx1 > 0))

  dist2, idx2 = pairwise_min_distance(segs, points, 10.0)
  print(dist2[idx2 > 0].max(), torch.sum(idx2 > 0))

  print(torch.all(idx1 == idx2))

  # print("Query grid...")
  # for i in tqdm(range(100)):
  #   grid.point_query(points, 5.0)


  skel = render.line_set(skeleton.points, skeleton.edges)
  o3d.visualization.draw([skel, boxes.render(), render.point_cloud(points)])

