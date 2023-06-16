from functools import cached_property
from numbers import Number
from typing import Optional
import numpy as np

from open3d_vis import render

from tensorclass import TensorClass
from dataclasses import dataclass

import torch
from .typecheck import typechecked, Float32, Int32,\
  Bool, jaxtyped, Vec1, Vec2, Vec3, NVec3, NFloat32





@dataclass(repr=False)
class Sphere(TensorClass):
  center: Vec3
  radius: Vec1

  def render(self, colors=None):
    return render.spheres(self.center, self.radius, colors=colors) 

  @property
  def bounds(self):
    return AABox(self.center - self.radius, self.center + self.radius)

  @jaxtyped
  def translate(self, d:Vec3):
    return Sphere(self.center + d, radius=self.radius)


@dataclass(repr=False)
class Point(TensorClass):
  p: Vec3

  def render(self, colors=None):
    return render.point_cloud(self.p, colors=colors) 

  @property
  def bounds(self):
    return AABox(self.p, self.p)

  def translate(self, d:Vec3):
    return Point(self.p + d)
  

@dataclass(repr=False)
class AABox(TensorClass):
  """An axis aligned bounding box in 3D space."""
  lower: Vec3
  upper: Vec3 

  @staticmethod
  def from_to(lower:Number | tuple[Number, Number, Number], upper:Number | tuple[Number, Number, Number], device:torch.device=torch.device('cpu')):
    if isinstance(lower, Number):
      lower = (lower, lower, lower)
    if isinstance(upper, Number):
      upper = (upper, upper, upper)

    lower = torch.tensor(lower, dtype=torch.float32, device=device)
    upper = torch.tensor(upper,  dtype=torch.float32, device=device)
    return AABox(lower, upper)

  @typechecked
  @staticmethod
  def from_points(points:Float32[torch.Tensor, '*M N 3']):
    return AABox(points.min(axis=-2).values, points.max(axis=-2).values)

  def expand(self, d:float):
    return AABox(self.lower - d, self.upper + d)

  @property
  def extents(self):
    return self.upper - self.lower

  def render(self, colors=None):
    return render.boxes(self.lower, self.upper, colors=colors)

  def random_points(self, n:int):
    return torch.rand(n, 3, device=self.device) * self.extents + self.lower

  @typechecked
  def clamp(self, points:NVec3):
    return torch.clamp(points, self.lower, self.upper)
  


  @typechecked
  def translate(self, d:Vec3):
    return AABox(self.lower + d, self.upper + d)

  def union(self, other:'AABox'):
    assert self.shape == other.shape, \
      f"Cannot union boxes of different shapes: {self.shape} != {other.shape}"

    return AABox(
      self.lower.min(other.lower),
      self.upper.max(other.upper)
    )
  
  def union_all(self) -> 'AABox':
    return AABox(
      self.lower.view(-1, 3).min(axis=0).values, 
      self.upper.view(-1, 3).max(axis=0).values
    )


  def merge(self): 
    return AABox(
      self.lower.view(-1, 3).min(axis=0).values,
      self.upper.view(-1, 3).max(axis=0).values
    )


@typechecked
def voxel_grid(lower:Vec3, upper:Vec3, voxel_size:float) -> AABox:
  extents = upper - lower
  grid_size = np.ceil(extents / voxel_size).to(torch.long)

  x, y, z = torch.meshgrid(*[torch.arange(x) for x in grid_size], indexing='ij')
  xyz = torch.stack([x, y, z], axis=-1).reshape(-1, 3)

  offset = lower + (extents - grid_size * voxel_size) / 2

  return AABox(
    lower = (offset + xyz * voxel_size),
    upper = (offset + (xyz + 1) * voxel_size),
  )

@dataclass(repr=False)
class Hit(TensorClass):
  seg : 'Segment'
  t1 : Vec1
  t2 : Vec1

  @property
  def valid(self) -> Bool[torch.Tensor, '*N']:
    return (self.t1 <= self.t2) & (self.t1 <= 1) & (self.t2 >= 0)

  @cached_property
  def p1(self):
    return self.seg.a + self.t1 * self.seg.dir

  @cached_property
  def p2(self):
    return self.seg.a + self.t1 * self.seg.dir



def dot(a, b):
  return torch.einsum('...d,...d->...', a, b)


@dataclass(repr=False)
class Plane(TensorClass):
  normal: Vec3
  d: Vec1

  def distance(self, p:Vec3):
    return dot(self.normal, p) + self.d
  
  @staticmethod
  def from_points(p1:Vec3, p2:Vec3, p3:Vec3):
    n = torch.cross(p2 - p1, p3 - p1)
    n = n / torch.norm(n)
    return Plane(n, -dot(n, p1))
  


@dataclass(repr=False)
class Line(TensorClass):
  p: Vec3
  dir: Vec3

  @typechecked
  def translate(self, d:Vec3):
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

@dataclass(repr=False)
class Segment(TensorClass):
  """Line segment between two points."""
  a: Vec3
  b: Vec3 

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
  
  def translate(self, d:Vec3):
    return Segment(self.a + d, self.b + d)

  
  def segment_closest(self:'Segment', seg2:'Segment'):
    s, t = self.line.line_closest(seg2.line)
    return torch.clamp(s, 0, 1), torch.clamp(t, 0, 1)

  def segment_distance(self, seg2:'Segment'):
    s, t = self.segment_closest(seg2)
    p1, p2 = self.points_at(s), seg2.points_at(t)

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
  
  def plane_intersections(seg:'Segment', plane:Plane) -> Hit:
      denom = dot(seg.dir, plane.normal)
      t = (plane.d - dot(seg.a, plane.normal)) / denom
      return Hit(seg, t, t)
  
  def render(self, colors=None):
    return render.segments(self.a, self.b, colors=colors)

  def points_at(self, t:NFloat32):
    return self.a + (self.b - self.a) * t.unsqueeze(-1)



@dataclass(repr=False)
class Tube(TensorClass):
  """Cylinder-like segment with radius varying linearly from one end to the other."""
  segment: Segment
  radii: Vec2

  def radius_at(self, t):
    return self.radii[:, 0] + (self.radii[:, 1] - self.radii[:, 0]) * t

  def translate(self, d:Vec3):
    return Tube(self.segment.translate(d), self.radii)

  @property
  def a(self):
    return Sphere(self.segment.a, self.radii[..., 0:1])

  @property
  def b(self):
    return Sphere(self.segment.b, self.radii[..., 1:2])

  @property
  def bounds(self):
    return self.a.bounds.union(self.b.bounds)
  
  def render(self, colors=None):
    flat = self.reshape(-1)
    def to_mesh(tube:Tube):
      return render.tube_mesh(points = torch.stack([tube.segment.a, tube.segment.b]), 
            radii=tube.radii, n=10)

    return render.concat_mesh([to_mesh(flat[i]) for i in range(flat.batch_shape[0])])

if __name__=="__main__":
  seg1 = Segment(torch.tensor([[0.0, -1.0, 0.0]]), torch.tensor([[0.0, 1.0, 0.0]]), convert_types=True)
  seg2 = Segment(torch.tensor([[0.0, 0.0, -1.0]]), torch.tensor([[1.0, 0.0, 1.0]]), convert_types=True)
  
  print(seg1)
  d = seg1.segment_distance(seg2)
  print(d)
  
  seg1 = Segment(torch.tensor([[0.0, 0.1, 0.0]]), torch.tensor([[0.0, 1.0, 0.0]]), convert_types=True)
  seg2 = Segment(torch.tensor([[0.0, 0.0, 0.1]]), torch.tensor([[0.0, 0.0, 1.0]]), convert_types=True)

  d = seg1.line.line_closest(seg2.line)
  print(d)

  points = torch.randn(17, 10, 3)
  box = AABox.from_points(points)

  print(box.shape, box.batch_shape)