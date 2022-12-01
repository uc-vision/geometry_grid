from dataclasses import dataclass
from typing import Any
import taichi as ti

import numpy as np

from .types import GridBounds
import geometry.np as np_geom

from nptyping import NDArray, Shape
from typeguard import typechecked

from .conversion import from_numpy


@ti.data_oriented
class Skeleton:
  @typechecked
  def __init__(self, vertices:ti.field, edges:ti.field, radii:ti.field):
    self.vertices = vertices
    self.edges = edges
    self.radii = radii
  

  @staticmethod
  def from_numpy(skeleton:np_geom.Skeleton):
    vertices = from_numpy(skeleton.points.astype(np.float32))
    edges = from_numpy(skeleton.edges, dtype=ti.i32)
    radii = from_numpy(skeleton.radii.astype(np.float32))

    return Skeleton(vertices, edges, radii)

  @ti.kernel
  def _box_intersection(min:ti.types.ndarray(), max:ti.types.ndarray(), counts:ti.types.ndarray()):




