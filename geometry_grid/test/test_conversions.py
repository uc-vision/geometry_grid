
from geometry_grid.taichi import geometry_types, conversion
from geometry_grid.torch import data_types

import taichi as ti
import torch

if __name__ == '__main__':
  ti.init()


  x = torch.randn(10, 17, 3, dtype=torch.float32)
  boxes = data_types.AABox.from_points(x)


  f = conversion.tensorclass_field(boxes, geometry_types.AABox)
  # f = ti.field(dtype=ti.f32, shape=(10, 17, 6))


  boxes2 = data_types.AABox(**f.to_torch(device=torch.device('cpu')))
  
  