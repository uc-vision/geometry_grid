
from typing import Tuple
import torch

import geometry.torch as torch_geom
from typeguard import typechecked


@typechecked
def around_segments(segments:torch_geom.Segment, radius:float, n:int):
  i = torch.randint(low=0, high=segments.shape[0] -1, 
    size=(int(n),), device=segments.device)

  t = torch.rand(n, device=segments.device) 
  r = torch.rand(n, 3, device=segments.device) * 2.0  - 1.0
  return segments[i].points(t) + r * radius

def random_segments(bounds:torch_geom.AABox, 
  length_range:Tuple[float, float]=(0.5, 2.0), n:int=100):

  lengths = torch.rand(n, device=bounds.device) * \
    (length_range[1] - length_range[0]) + length_range[0]

  points = bounds.random_points(n)
  directions = torch.rand(n, 3, device=bounds.device) * 2.0 - 1.0
  directions = directions / torch.norm(directions, dim=1, keepdim=True)

  return torch_geom.Segment(points, 
    bounds.clamp(points + directions * lengths.unsqueeze(1)))
