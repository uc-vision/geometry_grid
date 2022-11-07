from dataclasses import dataclass
from functools import cached_property
import drjit as dr
from drjit.cuda.ad import (Int, Float, 
  UInt32, Array3f, Array3i, TensorXf, Loop)


dr.set_log_level(dr.LogLevel.Info)

@dataclass
class Segment:
  a: Array3f
  b: Array3f


  @cached_property
  def dir(self):
    return self.b - self.a

  @cached_property
  def length_sq(self):
    d = self.dir
    return dr.dot(d, d)

  @cached_property
  def length(self):
    return dr.sqrt(self.length_sq)



@dataclass 
class Tubelet:
  seg: Segment
  r1: Float
  r2: Float
  
@dataclass 
class AABox:
  min: Array3f
  max: Array3f

