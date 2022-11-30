
from dataclasses import dataclass
import taichi as ti
import numpy as np

from nptyping import NDArray, Float, Int, Shape


Vec3 = NDArray[Shape["3"], Float]
IVec3 = NDArray[Shape["3"], Int]

@dataclass 
class Skeleton: 
  points: NDArray[Shape["N, 3"], Float]  
  radii: NDArray[Shape["N, 3"], Float]  
  edges: NDArray[Shape["M, 3"], Float]  


@dataclass
class AABox:
  """An axis aligned bounding box in 3D space."""

  min: Vec3
  max: Vec3

  def expand(self, d:float):
    return AABox(self.min - d, self.max + d)

@dataclass
class GridBounds:
  bounds : AABox
  size: IVec3
