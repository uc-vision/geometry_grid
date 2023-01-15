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
    
    return Tube(segment=self.segments, radii=radii)

@dataclass 
class Sphere(TensorClass):
  center: TensorType[3, float]
  radius: TensorType[float]

  def render(self, colors=None):
    return render.spheres(self.center, self.radius, colors=colors) 

  @property
  def bounds(self):
    return AABox(self.center - self.radius, self.center + self.radius)

  def translate(self, d:TensorType[3, float]):
    return Sphere(self.center + d, radius=self.radius)


@dataclass 
class Point(TensorClass):
  p: TensorType[3, float]

  def render(self, colors=None):
    return render.point_cloud(self.p, colors=colors) 

  @property
  def bounds(self):
    return AABox(self.p, self.p)

  def translate(self, d:TensorType[3, float]):
    return Point(self.p + d)


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

  def random_points(self, n:int):
    return torch.rand(n, 3, device=self.device) * self.extents + self.lower

  def clamp(self, points:TensorType[..., 3, torch.float32]):
    return torch.clamp(points, self.lower, self.upper)

  def translate(self, d:TensorType[3, float]):
    return AABox(self.lower + d, self.upper + d)

  def union(self, other:'AABox'):
    assert self.shape == other.shape, \
      f"Cannot union boxes of different shapes: {self.shape} != {other.shape}"

    return AABox(
      self.lower.min(other.lower),
      self.upper.max(other.upper)
    )


  def merge(self): 
    return AABox(
      self.lower.view(-1, 3).min(axis=0).values,
      self.upper.view(-1, 3).max(axis=0).values
    )


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
class Hit(TensorClass):
  seg : 'Segment'
  t1 : TensorType[float]
  t2 : TensorType[float]

  @property
  def valid(self) -> TensorType[..., bool]:
    return (self.t1 <= self.t2) & (self.t1 <= 1) & (self.t2 >= 0)

  @cached_property
  def p1(self):
    return self.seg.a + self.t1 * self.seg.dir

  @cached_property
  def p2(self):
    return self.seg.a + self.t1 * self.seg.dir





def dot(a, b):
  return torch.einsum('...d,...d->...', a, b)

@dataclass()
class Line(TensorClass):
  p: TensorType[3, float]
  dir: TensorType[3, float] 

  def translate(self, d:TensorType[3, float]):
    return Line(self.p + d, self.dir)

  def line_closest(line1:'Line', line2:'Line'):    
    v21 = line2.p - line1.p
      
    proj11 = dot(line1.dir, line1.dir)
    proj22 = dot(line2.dir, line2.dir)

    proj21 = dot(line2.dir, line1.dir)
    proj21_1 = dot(v21, line1.dir)
    proj21_2 = dot(v21, line2.dir)

    denom = proj21 * proj21 - proj22 * proj11

    s1 = denom.new_zeros(denom.shape)
    t1 = proj21_1 / proj21

    print(v21, proj21_1)

    s2 = (proj21_2 * proj21 - proj22 * proj21_1) / denom
    t2 = (-proj21_1 * proj21 + proj11 * proj21_2) / denom

    nz = torch.isclose(denom, torch.zeros(denom.shape))
    s = torch.where(nz, s1, s2)
    t = torch.where(nz, t1, t2)

    return s, t


@dataclass()
class Segment(TensorClass):
  """Line segment between two points."""
  a: TensorType[3, float]
  b: TensorType[3, float] 

  @property
  def length(self):
    return torch.norm(self.dir, dim=-1)

  @cached_property
  def dir(self):
    return self.b - self.a

  @cached_property
  def unit_dir(self):
    return self.dir / (self.length.unsqueeze(-1) + 1e-8)

  @cached_property
  def line(self):
    return Line(self.a, self.dir)

  @property
  def bounds(self):
    return AABox(self.a, self.b)
  
  def translate(self, d:TensorType[3, float]):
    return Segment(self.a + d, self.b + d)

  
  def segment_closest(self:'Segment', seg2:'Segment'):
    s, t = self.line.line_closest(seg2.line)
    return torch.clamp(s, 0, 1), torch.clamp(t, 0, 1)

  def segment_distance(self, seg2:'Segment'):
    s, t = self.segment_closest(seg2)
    p1, p2 = self.points(s), seg2.points(t)

    return torch.norm(p1 - p2, dim=-1)

  def box_intersections(seg:'Segment', bounds:AABox):
    dir = seg.b - seg.a

    a_start = (bounds.lower - seg.a) / dir
    a_end = (bounds.upper - seg.a) / dir 

    b_start = (seg.b - bounds.lower) / dir
    b_end = (seg.b - bounds.upper) / dir 

    t1 = torch.minimum(a_start, a_end).max(dim=2).values
    t2 = 1 - torch.minimum(b_start, b_end).max(dim=2).values

    return Hit(seg, t1, t2)

  def points_at(self, t:TensorType[..., float]):
    return self.a + (self.b - self.a) * t.unsqueeze(-1)




@dataclass
class Tube(TensorClass):
  """Line segment between two points."""
  segment: Segment
  radii: TensorType[2, float]

  def radius_at(self, t):
    return self.radii[:, 0] + (self.radii[:, 1] - self.radii[:, 0]) * t

  def translate(self, d:TensorType[3, float]):
    return Tube(self.segment.translate(d), self.radii)

  @property
  def a(self):
    return Sphere(self.segment.a, self.radii[:, 0])

  @property
  def b(self):
    return Sphere(self.segment.b, self.radii[:, 0])


  @property
  def bounds(self):
    return self.a.bounds.union(self.b.bounds,
      self.b.bounds).union(
        self.segment.bounds)

if __name__=="__main__":
  # seg1 = Segment(torch.tensor([[0.0, -1.0, 0.0]]), torch.tensor([[0.0, 1.0, 0.0]]), convert_types=True)
  # seg2 = Segment(torch.tensor([[0.0, 0.0, -1.0]]), torch.tensor([[1.0, 0.0, 1.0]]), convert_types=True)

  
  # d = seg1.segment_distance(seg2)
  # print(d)
  
  seg1 = Segment(torch.tensor([[0.0, 0.1, 0.0]]), torch.tensor([[0.0, 1.0, 0.0]]), convert_types=True)
  seg2 = Segment(torch.tensor([[0.0, 0.0, 0.1]]), torch.tensor([[0.0, 0.0, 1.0]]), convert_types=True)

  d = seg1.line.line_closest(seg2.line)
  print(d)