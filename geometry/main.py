

from pathlib import Path


import taichi as ti
import numpy as np
import torch
from geometry.torch.loading import display_skeleton, load_tree

from geometry.taichi.skeleton import BoxIntersection
from open3d_vis import render
import open3d as o3d

from geometry.torch.types import voxel_grid


ti.init(arch=ti.cuda, debug=True)



if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("filename", type=Path)
  args = parser.parse_args()

  skeleton = load_tree(args.filename, radius_threshold=10.0)

  bounds = skeleton.bounds
  boxes = voxel_grid(bounds.lower, bounds.upper, 100.0)
  # s = BoxIntersection.from_numpy(skeleton, boxes, max_intersections=10)

  # s.compute()

  # idx = np.flatnonzero(s.n_box.to_numpy())
  
  skel = render.line_set(skeleton.points, skeleton.edges)
  o3d.visualization.draw(skel)
  # o3d.visualization.draw([skel, boxes[idx].render()])

