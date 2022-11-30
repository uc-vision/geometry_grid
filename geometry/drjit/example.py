from typing import Tuple

import drjit as dr
import matplotlib.pyplot as plt
from drjit.cuda.ad import Array3f, Float, Loop, TensorXf, UInt32

from geometry_types import AABox, Segment, Tubelet
from segment import point_segment_dist

dr.set_log_level(dr.LogLevel.Info)



def sdf_tubelet(p:Array3f, tubelet:Tubelet) -> Float:
  t, dist_sq = point_segment_dist(p, tubelet.seg)
  r = dr.lerp(tubelet.r1, tubelet.r2, t)
  return dr.sqrt(dist_sq) - r  


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


img_t = TensorXf(dr.ravel(img), shape=(1500, 1500, 3))


plt.imshow(img_t)
plt.show()