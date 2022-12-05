# from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import numpy as np

from open3d_vis import render
from .dataclass import TensorClass, dataclass
from typeguard import typechecked

from torchtyping import TensorType
import torch

@dataclass 
class Skeleton: 


  points: TensorType['N', 3, float]  
  radii: TensorType['N', 1, float]
  edges: TensorType['M', 2, torch.long]  

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
    
    return Tube(segments=self.segments, radii=radii)


@dataclass
class AABox(TensorClass):
  """An axis aligned bounding box in 3D space."""
  lower: TensorType[3, float]
  upper: TensorType[3, float] 

  def expand(self, d:float):
    return AABox(self.lower - d, self.upper + d)

  @property
  def extents(self):
    return self.upper - self.lower

  def render(self, colors=None):
    return render.boxes(self.lower, self.upper, colors=colors)



@typechecked
def voxel_grid(lower:TensorType[3], upper:TensorType[3], voxel_size:float) -> AABox:
  extents = upper - lower
  grid_size = np.ceil(extents / voxel_size).to(torch.long)

  x, y, z = torch.meshgrid(*[torch.arange(x) for x in grid_size], indexing='ij')
  xyz = torch.stack([x, y, z], axis=-1).reshape(-1, 3)

  offset = lower + (extents - grid_size * voxel_size) / 2

  return AABox(
    lower = (offset + xyz * voxel_size),
    upper = (offset + (xyz + 1) * voxel_size),
  )




@dataclass()
class Segment(TensorClass):
  """Line segment between two points."""
  a: TensorType[3, float]
  b: TensorType[3, float] 

  @property
  def length(self):
    return np.linalg.norm(self.b - self.a, axis=-1)




@dataclass
class Tube(TensorClass):
  """Line segment between two points."""
  segments: Segment
  radii: TensorType[2, float]





