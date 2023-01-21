
import dataclasses
import taichi as ti
from taichi.math import vec3, ivec3, vec2
import taichi.math as tm

import numpy as np
import geometry_grid.torch as ng

@ti.dataclass
class Sphere:
    center: vec3
    radius: ti.f32

    @ti.func
    def area(self):
        # a function to run in taichi scope
        return 4 * np.pi * self.radius * self.radius

    @ti.func
    def intersects_box(self, box:ti.template()):
      box.point_distance(self.center) <= self.radius

    @ti.func
    def bounds(self):
      return AABox(self.center - self.radius, self.center + self.radius)

    @ti.func 
    def point_distance(self, p:vec3):
      return (p - self.center).norm() - self.radius



@ti.dataclass
class Point:
    p: vec3

    @ti.func
    def intersects_box(self, box:ti.template()):
      return box.contains(self.p)

    @ti.func
    def bounds(self):
      return AABox(self.p, self.p)

    @ti.func 
    def point_distance(self, p:vec3):
      return (p - self.p).norm()


bvec3 = ti.types.vector(3, bool)

@ti.dataclass
class AABox:
  """An axis aligned bounding box in 3D space."""

  lower: vec3
  upper: vec3

  @ti.func
  def expand(self, d:ti.f32):
    return AABox(self.lower - d, self.upper + d)

  @ti.func 
  def center(self):
    return (self.lower + self.upper) / 2

  @ti.func
  def extents(self):
    return self.upper - self.lower

  @ti.func
  def union(self, box:ti.template()):
    return AABox(
        ti.min(self.lower, box.lower), 
        ti.max(self.upper, box.upper))

  @ti.func
  def contains(self, p:vec3):
    return (p >= self.lower).all() and (p <= self.upper).all()


  @ti.func 
  def edges(self):
    l = self.lower
    u = self.upper
    # There are 12 edges
    return (
      Segment(l, vec3(u.x, l.y, l.z)),
      Segment(l, vec3(l.x, u.y, l.z)),
      Segment(l, vec3(l.x, l.y, u.z)),
      Segment(vec3(u.x, u.y, l.z), u),
      Segment(vec3(u.x, l.y, u.z), u),
      Segment(vec3(l.x, u.y, u.z), u),
      Segment(vec3(u.x, l.y, l.z), vec3(u.x, u.y, l.z)),
      Segment(vec3(u.x, l.y, l.z), vec3(u.x, l.y, u.z)),
      Segment(vec3(l.x, u.y, l.z), vec3(u.x, u.y, l.z)),
      Segment(vec3(l.x, u.y, l.z), vec3(l.x, u.y, u.z)),
      Segment(vec3(l.x, l.y, u.z), vec3(u.x, l.y, u.z)),
      Segment(vec3(l.x, l.y, u.z), vec3(l.x, u.y, u.z)),
    )

  @ti.func 
  def distance(self, p:vec3):
    d = vec3(0.)
    for i in ti.static(range(3)):
      d[i] +=  ti.max(0., p[i] - self.lower[i]) + ti.max(0., p[i] - self.upper[i])
    return ti.sqrt(tm.dot(d, d))

@ti.dataclass
class Segment:
  """A line segment in 3D space."""

  a: vec3
  b: vec3

  @ti.func
  def dir(self) -> vec3:
    return self.b - self.a

  @ti.func
  def point_at(self, t:ti.f32) -> vec3:
    return self.a + t * self.dir()

  @ti.func
  def line(self):
    return Line(self.a, self.dir())

  @ti.func
  def length_sq(self):
    d = self.dir()
    return ti.dot(d, d)

  @ti.func
  def length(self):
    return ti.sqrt(self.length_sq())

  @ti.func
  def bounds(self):
    return AABox(ti.min(self.a, self.b), ti.max(self.a, self.b))

  @ti.func
  def point_dist_sq(self, p:vec3, eps=1e-6):
    d = self.dir()
    l2 = tm.dot(d, d)  # |b - a|^2

    t = tm.clamp(tm.dot(d, (p - self.a) / l2), 0., 1.)
    p_proj = self.a + t * d
    delta_p = p_proj - p

    dist_sq = tm.dot(delta_p, delta_p)
    pb = p - self.b

    dist_sq = ti.select(l2 >= eps, dist_sq, tm.dot(pb, pb))
    return t, dist_sq

  @ti.func
  def point_distance(self, p:vec3):
    t, dist_sq = self.point_dist_sq(p)
    return ti.sqrt(dist_sq)

  @ti.func
  def segment_closest(seg1, seg2:ti.template()) -> vec2:
    line2 = seg2.line()
    line1 = seg1.line()

    t = line1.line_closest(line2)
    return tm.clamp(t, 0., 1.)

  @ti.func
  def segment_distance(seg1, seg2:ti.template()) -> ti.f32:
    t = seg1.segment_closest(seg2)
    return tm.distance(seg1.point_at(t[0]), seg2.point_at(t[1]))


  @ti.func
  def box_intersections(self, box:ti.template()) -> vec2:
    d = self.dir() + 1e-8
    
    a_start = (box.lower - self.a) / d
    a_end = (box.upper - self.a) / d 

    b_start = (self.b - box.lower) / d
    b_end = (self.b - box.upper) / d 


    return  vec2(
      tm.min(a_start, a_end).max(),  
      1 - tm.min(b_start, b_end).max()
    )
    

  @ti.func
  def box_distance(self, box:ti.template()):
    ds = [self.segment_distance(e) for e in ti.static(box.edges())]

    d1 = box.distance(self.a)
    d2 = box.distance(self.b)

    return ti.select(self.intersects_box(box), 0, ti.min(*ds, d1, d2))

  @ti.func
  def intersects_box(self, box:ti.template()):
    i = self.box_intersections(box)
    return i[0] <= i[1] and i[0] <= 1 and i[1] >= 0





@ti.dataclass
class Line:
  p: vec3
  dir: vec3

  @ti.func
  def line_closest(line1, line2:ti.template(), eps=1e-8) -> vec2:    
    v21 = line2.p - line1.p
      
    proj11 = tm.dot(line1.dir, line1.dir)
    proj22 = tm.dot(line2.dir, line2.dir)

    proj21 = tm.dot(line2.dir, line1.dir)
    proj21_1 = tm.dot(v21, line1.dir)
    proj21_2 = tm.dot(v21, line2.dir)

    denom = proj21 * proj21 - proj22 * proj11

    s1 = 0.0
    t1 = proj21_1 / (proj21 + eps)

    s2 = (proj21_2 * proj21 - proj22 * proj21_1) / denom
    t2 = (-proj21_1 * proj21 + proj11 * proj21_2) / denom

    s = ti.select(denom > eps, s1, s2)
    t = ti.select(denom > eps, t1, t2)

    return vec2(s, t)




@ti.dataclass
class Tube:
  """ A tube is a capsule with radius 
  linearly changing along the segment. 
  """
  
  segment: Segment
  radii: vec2

  @ti.func
  def radius_at(self, t:ti.f32):
    return self.r[0] + t * (self.r[1]- self.r[0])


  def point_distance(self, p:vec3):
    t, dist_sq = self.segment.point_dist_sq(p)
    r = self.radius_at(t)
    return ti.sqrt(dist_sq) - r


  @ti.func
  def bounds(self):
    r1, r2 = self.radii
    b1 =  AABox(self.segment.a - r1, self.segment.a + r1)
    b2 =  AABox(self.segment.b - r2, self.segment.b + r2)

    return b1.union(b2)


  @ti.func
  def radius_at(self, t:ti.f32):
    return self.radii[0] + t * (self.radii[1]- self.radii[0])

  @ti.func
  def segment_distance(self, segment:ti.template()) -> ti.f32:
    t = self.segment.segment_closest(segment)
    d = tm.distance(self.segment.point_at(t[0]), segment.point_at(t[1]))

    r = self.radius_at(t[0])
    return d - r
    
  @ti.func
  def intersects_box(self, box:ti.template()):
    if self.segment.box_intersections(box):
      return True

    for e in ti.static(box.edges()):
      if self.segment_distance(e) <= 0.:
        return True

    d1 = box.distance(self.segment.a)
    d2 = box.distance(self.segment.b)

    return d1 < self.r[0] or d2 < self.r[1]
  

  def approx_intersects_box(self, box:AABox):
    r = ti.max([self.r1, self.r2])
    box = box.expand(r)
    return self.seg.intersects_box(box)






if __name__ == '__main__':
  
  ti.init(arch=ti.cpu, debug=True)

  box = AABox(vec3(-0.5, -0.5, -0.5), vec3(0.5, 0.5, 0.5))
  seg = Segment(vec3(-2, 0, 0), vec3(0, 2, 0))




  @ti.kernel
  def test_seg_box() -> vec2:
    return seg.box_intersections(box)

  print(test_seg_box())
  

  seg1 = Segment(vec3(0.0, -0.5, 0.0), vec3(0.0, 1.0, 0.0))
  seg2 = Segment(vec3(0.1, 0.0, -0.5), vec3(0.1, 0.0, 1.0))


  @ti.kernel
  def test_line_line() -> vec2:
    line1 = seg1.line()
    line2 = seg2.line()
    return line1.line_closest(line2)

  print(test_line_line())

 
  @ti.kernel
  def test_seg_seg() -> ti.f32:
    return seg1.segment_distance(seg2)

  print(test_seg_seg())
