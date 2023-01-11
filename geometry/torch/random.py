
from typing import Tuple
import torch

import geometry.torch as torch_geom
from typeguard import typechecked

from open3d_vis import render
import open3d as o3d

from geometry.torch.types import dot


@typechecked
def around_tubes(tubes:torch_geom.Tube, n:int, point_var:float = 0.01):
  i = torch.randint(low=0, high=tubes.shape[0] -1, 
    size=(int(n),), device=tubes.device)

  tubes = tubes[i]
  segments = tubes.segment

  t = torch.rand((n,), device=segments.device) 
  p = torch.randn(n, 3, device=segments.device) * point_var

  d = segments.unit_dir

  v = torch.randn( (n, 3), device=segments.device)
  v = v - dot(v, d).unsqueeze(-1) * d

  r = v / torch.norm(v, dim=1, keepdim=True)
  return segments.points_at(t) + r * tubes.radius_at(t).unsqueeze(1) + p

  

def random_segments(bounds:torch_geom.AABox, 
  length_range:Tuple[float, float]=(0.5, 2.0), n:int=100):

  lengths = torch.rand(n, device=bounds.device) * \
    (length_range[1] - length_range[0]) + length_range[0]

  points = bounds.random_points(n)
  directions = torch.rand(n, 3, device=bounds.device) * 2.0 - 1.0
  directions = directions / torch.norm(directions, dim=1, keepdim=True)

  return torch_geom.Segment(points, 
    bounds.clamp(points + directions * lengths.unsqueeze(1)))
