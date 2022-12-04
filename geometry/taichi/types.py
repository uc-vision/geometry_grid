
import dataclasses
import taichi as ti
from taichi.math import vec3, ivec3
import taichi.math as tm

import numpy as np
import geometry.torch as ng

@ti.dataclass
class Sphere:
    center: vec3
    radius: ti.f32

    @ti.func
    def area(self):
        # a function to run in taichi scope
        return 4 * np.pi * self.radius * self.radius

@ti.dataclass
class AABox:
  """An axis aligned bounding box in 3D space."""

  min: vec3
  max: vec3

  @ti.func
  def expand(self, d:ti.f32):
    return AABox(self.min - d, self.max + d)

  @ti.func
  def contains(self, p:vec3):
    for i in ti.static(range(3)):
      if p[i] < self.min[i] or p[i] > self.max[i]:
        return False
        
    return True




@ti.dataclass
class Segment:
  """A line segment in 3D space."""

  a: vec3
  b: vec3

  @ti.func
  def dir(self):
    return self.b - self.a

  @ti.func
  def length_sq(self):
    d = self.dir()
    return ti.dot(d, d)

  @ti.func
  def length(self):
    return ti.sqrt(self.length_sq())

  @ti.func
  def point_dist_sq(self, p:vec3, eps=1e-6):
    d = self.dir()
    l2 = tm.dot(d, d)  # |b - a|^2

    t = tm.clamp(tm.dot(d, (p - self.a) / l2), 0., 1.)
    p_proj = self.a + t * d
    delta_p = p_proj - p

    dist_sq = tm.dot(delta_p, delta_p)
    pb = p - self.b

    dist_sq = ti.select(l2 >= eps, dist_sq, ti.dot(pb, pb))
    return t, dist_sq

  @ti.func
  def sdf(self, p:vec3):
    t, dist_sq = self.point_dist_sq(p)
    return ti.sqrt(dist_sq)

  @ti.func
  def box_intersections(self, box:ti.template()):
    dir = self.dir()

    a_start = (box.min - self.a) / dir
    a_end = (box.max - self.a) / dir 

    b_start = (self.b - box.min) / dir
    b_end = (self.b - box.max) / dir 

    return  ti.math.vec2(tm.min(a_start, a_end).min(),  
      1 - tm.min(b_start, b_end).max())


  @ti.func
  def intersects_box(self, box:ti.template()):
    i = self.box_intersections(box)
    return i[0] <= i[1] and i[0] <= 1 and i[1] >= 0

@ti.dataclass
class Tubelet:
  """ A tubelet is a capsule with radius 
  linearly changing along the segment. 
  """
  
  seg: Segment
  r1: ti.f32
  r2: ti.f32

  @ti.func
  def radius_at(self, t:ti.f32):
    return self.r1 + t * (self.r2 - self.r1)


  def sdf(self, p:vec3):
    t, dist_sq = self.seg.point_dist_sq(p)
    r = self.radius_at(t)
    return ti.sqrt(dist_sq) - r
  
  def approx_intersects_box(self, box:AABox):
    r = ti.min([self.r1, self.r2])
    box = box.expand(r)

    return self.seg.intersects_box(box)
