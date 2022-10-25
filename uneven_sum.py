from dataclasses import dataclass
from typing import Tuple
import drjit as dr
from drjit.cuda.ad import (Int, Float, 
  UInt32, Array3f, Array3i, TensorXf, Loop)


def uneven_sum():
  f = dr.arange(Float, 0, 20, 1)
  offset = Float([0, 3, 11, 13])
  length =  Float([3, 8, 2, 6])


  sum = Float(0.0)
  i = UInt32(0)


  loop = Loop("Summation", lambda: (i, sum))
  while loop(i < length):

    sum += dr.gather(Float, f, offset + i)
    i += 1

  print(sum)


if __name__ == "__main__":
  uneven_sum()