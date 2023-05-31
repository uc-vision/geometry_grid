from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import numpy as np

from open3d_vis import render
import open3d as o3d

from py_structs.numpy import shape
import torch

from geometry_grid.taichi_geometry.geometry_types import Tube
from ..torch_geometry.geometry_types import AABox, Segment


from geometry_grid.torch_geometry.typecheck import Float32, Int32, jaxtyped

@jaxtyped
@dataclass 
class Skeleton: 

  points: Float32[torch.Tensor, 'N 3']  
  radii: Float32[torch.Tensor, 'N 1']
  edges: Int32[torch.Tensor, 'M 2']  

  @property
  def bounds(self):
    return AABox(self.points.min(axis=0).values, self.points.max(axis=0).values)

  @cached_property
  def segments(self):

    return Segment(
      self.points[self.edges[:, 0]], 
      self.points[self.edges[:, 1]])

  @cached_property
  def tubes(self):
    radii = torch.stack([
        self.radii[self.edges[:, 0]], 
        self.radii[self.edges[:, 1]]], dim=-1).squeeze(1)
    
    return Tube(segment=self.segments, radii=radii)
  



def load_tree(filename:Path, radius_threshold=3.0):
  skeleton =  np.load(filename, allow_pickle=True)

  v = skeleton['skeleton_verts']
  r = skeleton['skeleton_radii']

  edges = skeleton['skeleton_edges']
  valid = np.flatnonzero((r > radius_threshold).reshape(-1))

  radii = np.zeros( (v.shape[0], 1) ) 
  radii[edges[:, 1]] = r
  radii[edges[:, 0]] = r



  return Skeleton(
    torch.from_numpy(v).to(torch.float32),
    torch.from_numpy(radii).to(torch.float32),
    torch.from_numpy(edges[valid]).to(torch.long) )


def display_skeleton(skeleton:Skeleton):
  r = skeleton.radii[skeleton.edges[:, 1]]
  t = r / r.max()

  colors = torch.stack([1.0 - t, t, torch.zeros_like(t)], dim=1)
  return render.line_set(skeleton.points, skeleton.edges, colors=colors)





if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("filename", type=Path)
  args = parser.parse_args()


  skeleton = load_tree(args.filename)
  display_skeleton(skeleton)
  