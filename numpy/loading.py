from dataclasses import dataclass
from pathlib import Path
import numpy as np

from open3d_vis import render
import open3d as o3d

from py_structs.numpy import shape
from ..taichi.skeleton import Skeleton



def load_tree(filename:Path, radius_threshold=3.0):
  skeleton =  np.load(filename, allow_pickle=True)

  v = skeleton['skeleton_verts']
  r = skeleton['skeleton_radii']

  edges = skeleton['skeleton_edges']
  valid = (r > radius_threshold).reshape(-1)

  radii = np.zeros(v.shape) 
  radii[edges[:, 0]] = r

  return Skeleton(v, radii, edges[valid])

def display_skeleton(skeleton:Skeleton):

  print(shape(skeleton))

  skel = render.line_set(skeleton.points, skeleton.edges)
  o3d.visualization.draw(skel)




if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("filename", type=Path)
  args = parser.parse_args()


  skeleton = load_tree(args.filename)
  display_skeleton(skeleton)
  