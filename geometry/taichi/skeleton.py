from dataclasses import dataclass
import taichi as ti
import numpy as np

import geometry.np as np_geom

ti.init()




def from_numpy(x:np.ndarray, dt=ti.f32):
  v = ti.Vector.field(x.shape[-1], dt=dt)
  v.from_numpy(x)
  return v


@ti.data_oriented
class Skeleton:
  def __init__(self, vertices, edges, radii):
    self.vertices = vertices
    self.edges = edges
    self.radii = radii
  

  @staticmethod
  def from_numpy(skeleton:np_geom.Skeleton):
    vertices = from_numpy(skeleton.points)
    edges = from_numpy(skeleton.edges, dt=ti.i32)
    radii = from_numpy(skeleton.radii)
    return Skeleton(vertices, edges, radii)
