from dataclasses import dataclass, field
from functools import cached_property
from typing import Generic, TypeVar

import drjit as dr
from drjit.cuda.ad import Array3f, Float, Int

dr.set_log_level(dr.LogLevel.Info)

T = TypeVar("T")

@dataclass
class Segment:
  """A line segment in 3D space."""
  DRJIT_STRUCT  = { 'a' : Array3f, 'b' : Array3f }

  a: Array3f = field(default_factory=Array3f)
  b: Array3f = field(default_factory=Array3f)

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
  """ A tubelet is a capsule with radius 
  linearly changing along the segment. 
  """
  
  DRJIT_STRUCT  = { 'seg':Segment, 'r1' : Float, 'r2' : Float }

  seg: Segment = field(default_factory=Segment)
  r1: Float = field(default_factory=Float)
  r2: Float = field(default_factory=Float)
  
@dataclass 
class AABox:
  DRJIT_STRUCT  = { 'min':Array3f, 'max' : Array3f }

  min: Array3f = field(default_factory=Array3f)
  max: Array3f = field(default_factory=Array3f)



@dataclass 
class Ragged:
  DRJIT_STRUCT  = { 'offsets':Int, 'length' : Int }

  offset: Int = field(default_factory=Int)
  length: Int = field(default_factory=Int)


class RaggedArrays(Generic[T]):
  def __init__(self, data:T, ragged:Ragged):
    self.data = data
    self.ragged = ragged



  def __len__(self):
    return self.ragged.length

  def __iter__(self):
    for i in range(len(self)):
      yield self[i]
