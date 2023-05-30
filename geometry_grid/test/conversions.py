
from geometry_grid.taichi_geometry import conversion

import geometry_grid.taichi_geometry as ti_geom
import geometry_grid.torch_geometry as torch_geom

import taichi as ti
import torch




if __name__ == '__main__':
  ti.init()


  x = torch.randn(10, 17, 3, dtype=torch.float32)
  boxes = torch_geom.AABox.from_points(x)

  f = conversion.tensorclass_field(boxes, ti_geom.AABox)
  # f = ti.field(dtype=ti.f32, shape=(10, 17, 6))

  boxes2 = torch_geom.AABox(**f.to_torch())

