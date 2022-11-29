
from dataclasses import dataclass
import taichi as ti
import numpy as np



@dataclass 
class Skeleton: 
  points: np.ndarray  # float: N, 3 
  radii: np.ndarray # float: N, 1

  edges: np.ndarray  # int M, 2