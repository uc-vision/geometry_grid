

from pathlib import Path


import taichi as ti
import numpy as np
import torch
from geometry.taichi.conversion import from_torch, torch_field
from geometry.torch.loading import display_skeleton, load_tree

from geometry.taichi.skeleton import BoxIntersection, Grid
import geometry.taichi as ti_geom
import geometry.torch as torch_geom


from py_structs.torch import shape

from open3d_vis import render
import open3d as o3d

from geometry.torch.types import voxel_grid


ti.init(arch=ti.cuda, debug=True)

@ti.func
def point_bounds(points:ti.template(dim=1)) -> ti_geom.AABox:
  bounds = ti_geom.AABox(points[0], points[0])
  for i in points:
    bounds = bounds.union(ti_geom.AABox(points[i], points[i]))
  return bounds


@ti.kernel
def point_bounds_(points:ti.types.ndarray(dtype=ti.math.vec3), bounds:ti.template()):
  bounds[0] = point_bounds(points)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("filename", type=Path)
  args = parser.parse_args()

  skeleton = load_tree(args.filename, radius_threshold=10)

  grid = Grid.from_torch(skeleton.bounds, 8, 64)

  tubes = from_torch(skeleton.tubes.segment) 
  grid.intersect_dense(tubes)

  print(grid.get_counts())
  grid = grid.subdivided(tubes)

  print(grid.get_counts())


  # s = BoxIntersection.from_torch(skeleton, boxes, max_intersections=10)
  # s.compute()

  # box_hits = s.n_box.to_numpy()
  # box_idx = np.flatnonzero(box_hits > 0)

  # print(box_hits[box_idx])
  
  # hits = skeleton.segments.box_intersections(boxes)  
  # box_idx = torch.nonzero( torch.any(hits.valid, dim=1) ).squeeze()

  

  # skel = render.line_set(skeleton.points, skeleton.edges)
  # o3d.visualization.draw([skel, boxes[box_idx].render()])

