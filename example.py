from dataclasses import dataclass
from typing import Tuple
import drjit as dr
from drjit.cuda.ad import (Int, Float, 
  UInt32, Array3f, Array3i, TensorXf, Loop)

import matplotlib.pyplot as plt

dr.set_log_level(dr.LogLevel.Info)

@dataclass
class Segment:
  a: Array3f
  b: Array3f


@dataclass 
class Tubelet:
  seg: Segment
  r1: Float
  r2: Float
  
@dataclass 
class AABox:
  min: Array3f
  max: Array3f


@dataclass 
class VoxelGrid:
  bounds : AABox
  dim : Array3i

  entries: Int
  counts: Int 


def distance_segment(p:Array3f, seg:Segment, 
  eps:Float=1e-6) -> Tuple[Float, Float]:

  d = seg.b - seg.a
  l2 = dr.dot(d, d)  # |b - a|^2

  t = dr.clamp(dr.dot(d, (p - seg.a) / l2), 0., 1.)
  p_proj = seg.a + t * d
  delta_p = p_proj - p

  dist_sq = dr.dot(delta_p, delta_p)
  pb = p - seg.b

  dist_sq = dr.select(l2 >= eps, dist_sq, dr.dot(pb, pb))
  return t, dist_sq


def sdf_tubelet(p:Array3f, tubelet:Tubelet) -> Float:
  t, dist_sq = distance_segment(p, tubelet.seg)
  r = dr.lerp(tubelet.r1, tubelet.r2, t)
  return dr.sqrt(dist_sq) - r  


def distance_segment_box(box:AABox, seg:Segment):
  dir = seg.b - seg.a

  a_start = (box.min - seg.a) / dir
  a_end = (box.max - seg.a) / dir 

  b_start = (seg.b - box.min) / dir
  b_end = (seg.b - box.max) / dir 

  return  (dr.minimum(a_start, a_end),  
    1 - dr.minimum(b_start, b_end))




def sdf(p:Array3f) -> Float:

  seg = Segment(Array3f(-1, -1, 0), Array3f(1, 1, 0))
  tubelet = Tubelet(seg, 0.4, 0.2) 

  return sdf_tubelet(p, tubelet)
  
def trace(o: Array3f, d: Array3f) -> Array3f:
  i = UInt32(0)
  loop = Loop("Sphere tracing", lambda: (o, i))
  while loop(i < 20):
    x = sdf(o)

    o = dr.fma(d, x, o)
    i += 1
  return o






# def shade(p: Array3f, l: Array3f) -> Float:
#   dr.enable_grad(p)
#   value = sdf(p)

#   dr.set_grad(p, l)
#   dr.forward_to(value)
#   return dr.maximum(0, dr.grad(value))

def normal(p: Array3f) -> Array3f:
  dr.enable_grad(p)
  value = sdf(p)

  dr.backward(value)
  return dr.grad(p)




x = dr.linspace(Float, -1, 1, 1500)
x, y = dr.meshgrid(x, x)

p = trace(o=Array3f(0, 0, -2), d=dr.normalize(Array3f(x, y, 1)))
print(dr.shape(x), dr.width(x), dr.shape(y))

light_pos = Array3f(-4, 2, -8)
light_dir = dr.normalize(light_pos - p)


n = normal(p)
sh = dr.dot(n, light_dir) 


img = Array3f(1.0, 0.8, .0) * sh + Array3f(0, .0, .4) 
img[sdf(p) > .1] = 0


img_flat = dr.ravel(img)

img_t = TensorXf(img_flat, shape=(1500, 1500, 3))


plt.imshow(img_t)
plt.show()