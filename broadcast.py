
import drjit as dr
import numpy as np
from drjit.cuda.ad import Array3f, TensorXf, Arrayf64

from geometry_types import Segment
from segment import seg_seg_dist

dr.set_log_level(dr.LogLevel.Info)


def random_segments(n:int, rng=np.random.default_rng()) -> Segment:
  return Segment(random_vectors(n, rng), random_vectors(n, rng))

def random_vectors(n:int, rng=np.random.default_rng()) -> Array3f:
  a = rng.random((n, 3)) * 2.0 - 1.0
  return Array3f(a)


if __name__ == '__main__':
  dr.set_log_level(dr.LogLevel.Info)

  # x = random_segments(10)
  # y = random_segments(10)


  # d = seg_seg_dist(x, y)
  # print(d)

