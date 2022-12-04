import dataclasses
import taichi as ti
import numpy as np

from geometry.torch import Skeleton, AABox
from typeguard import typechecked

from geometry.taichi.types import AABox, Segment
from .conversion import from_numpy, to_field


@ti.dataclass
class Intersection:
  t : ti.f32
  i : ti.i32



@ti.data_oriented
class BoxIntersection:
  def __init__(self, segments:Segment, boxes:AABox, max_intersections=10):
    self.segments = segments
    self.boxes = boxes

    self.intersection = ti.field(ti.i32) # Intersection.field()
    box = ti.root.dense(ti.i, self.boxes.shape)
    hits = box.dynamic(ti.j, max_intersections, chunk_size=4)
    hits.place(self.intersection)

    self.n_box = ti.field(ti.i32, shape=self.boxes.shape)
    self.n_segment = ti.field(ti.i32, shape=self.segments.shape)

    self.n_box.fill(0)
    self.n_segment.fill(0)



  def from_numpy(skeleton:Skeleton, boxes:AABox, max_intersections=10): 
    print(skeleton.segments.size)   
    return BoxIntersection(
      segments=to_field(skeleton.segments, Segment),
      boxes=to_field(boxes, AABox),
      max_intersections=max_intersections)


  @ti.kernel
  def compute(self):

    for i in self.boxes:
      for j in range(self.segments.shape[0]):

        if self.segments[j].intersects_box(self.boxes[i]):
          self.intersection[i].append(j)
          
          self.n_box[i] += 1
          self.n_segment[j] += 1




