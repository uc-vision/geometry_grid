
import drjit as dr
from drjit.cuda.ad import Array3f, Float, Array3i, Int

from geometry_types import AABox

def grid_3d(x:int, y:int, z:int) -> AABox:
  x, y, z = dr.meshgrid(
      dr.arange(Int, start=0, stop=x),
      dr.arange(Int, start=0, stop=y),
      dr.arange(Int, start=0, stop=z),
  )

  return AABox(min=Array3f(x, y, z), max=Array3f(x + 1, y + 1, z + 1))






if __name__ == '__main__':
  dr.set_log_level(dr.LogLevel.Info)

  box = grid_3d(2, 2, 2)
  print(box, dr.width(box))