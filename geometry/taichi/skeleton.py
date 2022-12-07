import dataclasses
import taichi as ti
import numpy as np

from geometry import torch as torch_geom
from typeguard import typechecked

from geometry.taichi.types import AABox, Segment, Tube
from .conversion import from_torch, torch_field


@ti.dataclass
class Intersection:
  t : ti.f32
  i : ti.i32



@ti.data_oriented
class BoxIntersection:
  def __init__(self, tubes:Tube, boxes:AABox, max_intersections=10):
    self.tubes = tubes
    self.boxes = boxes

    self.intersection = ti.field(ti.i32) # Intersection.field()
    box = ti.root.dense(ti.i, self.boxes.shape)
    hits = box.dynamic(ti.j, max_intersections, chunk_size=4)
    hits.place(self.intersection)

    self.n_box = ti.field(ti.i32, shape=self.boxes.shape)
    self.n_tube = ti.field(ti.i32, shape=self.tubes.shape)



  def from_torch(skeleton:torch_geom.Skeleton, boxes:AABox, max_intersections=10): 
    return BoxIntersection(
      tubes=torch_field(skeleton.tubes, Tube),
      boxes=torch_field(boxes, AABox),
      max_intersections=max_intersections)


  @ti.kernel
  def compute(self):

    for i in self.boxes:
      for j in range(self.tubes.shape[0]):

        if self.tubes[j].intersects_box(self.boxes[i]):
          self.intersection[i].append(j)
          
          self.n_box[i] += 1
          self.n_tube[j] += 1




