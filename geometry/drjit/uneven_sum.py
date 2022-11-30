
import drjit as dr
from drjit.cuda.ad import Float, Loop, UInt32, Array3f, Int, TensorXf

import numpy as np

import torch


def uneven_sum():
  f = dr.arange(Float, 0, 20, 1)
  offset = Int([0, 3, 11, 13])
  length =  Int([3, 8, 2, 6])

  sum = Float(0.0)
  i = UInt32(0)

  loop = Loop("Summation", lambda: (i, sum))
  while loop(i < length):

    sum += dr.gather(Float, f, offset + i)
    i += 1

  print(sum)



if __name__ == "__main__":
  uneven_sum()
