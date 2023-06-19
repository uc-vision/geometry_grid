
from open3d_vis import render
import open3d as o3d

import torch

def display_distances(geom, boxes, points, dist):
  red = torch.tensor([1.0, 0.0, 0.0], device=points.device)
  green = torch.tensor([0.0, 1.0, 0.0], device=points.device)

  t = dist.unsqueeze(1).clamp(0.0, 1.0)

  colors = (red * t + green * (1.0 - t)).squeeze()
  geom = geom.render().paint_uniform_color((0, 0, 1))

  o3d.visualization.draw([
    dict(name="geometry", geometry=geom, material=None, group="", is_visible=True),
    dict(name="boxes", geometry=boxes.render(), material=None, group="", is_visible=False),
    dict(name="points", geometry=render.point_cloud(points, colors=colors), material=None, group="", is_visible=True),
  ], show_ui=True)
