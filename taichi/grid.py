from dataclasses import dataclass
import taichi as ti
import numpy as np

ti.init()

@dataclass 
class SegmentNp:
  a : np.ndarray
  b : np.ndarray

  
@dataclass 
class TubeletNp:
  segment : SegmentNp
  r1 : np.ndarray
  r2 : np.ndarray



@ti.data_oriented
class LineSet:
    def __init__(self, tubes_np):
        self.lines = ti.StructField(
          dict(a=)  
        )
        self.lines.from_numpy(lines)
        

    @ti.kernel
    def inc(self):
        for i in self.x:
            self.x[i] += 1

a = TiArray(32)
a.inc()