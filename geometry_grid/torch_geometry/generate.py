
from typing import Tuple
import torch

import geometry_grid.torch_geometry as torch_geom
from typeguard import typechecked

from open3d_vis import render
import open3d as o3d

from geometry_grid.torch_geometry.geometry_types import dot


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

  r = v / (torch.norm(v, dim=1, keepdim=True) + 1e-8)
  return segments.points_at(t) + r * tubes.radius_at(t).unsqueeze(1) + p

  
@typechecked
def around_segments(segments:torch_geom.Segment, n:int, radius:float, point_var:float = 0.01):
  tubes = torch_geom.Tube(segments, torch.full((segments.shape[0], 2), radius, device=segments.device))
  return around_tubes(tubes, n, point_var=point_var)

def random_segments(bounds:torch_geom.AABox,
  length_range:Tuple[float, float]=(0.5, 2.0), n:int=100):

  lengths = torch.rand(n, device=bounds.device) * \
    (length_range[1] - length_range[0]) + length_range[0]

  points = bounds.random_points(n)
  directions = torch.rand(n, 3, device=bounds.device) * 2.0 - 1.0
  directions = directions / torch.norm(directions, dim=1, keepdim=True)

  return torch_geom.Segment(points, 
    bounds.clamp(points + directions * lengths.unsqueeze(1)))



def random_tubes(segments:torch_geom.Segment, radius_range:Tuple[float, float]=(0.1, 0.25), n:int=100):
  radii = torch.rand(n, 2, device=segments.device) * radius_range[1] + radius_range[0]

  return torch_geom.Tube(segments, radii)

