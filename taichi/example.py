import taichi as ti
import taichi.math as tm

import numpy as np

ti.init()

x = np.linspace(0, 1, 100)
coords = np.stack(np.meshgrid(x, x), -1).astype(np.float32)


field = ti.Vector.field(n=3, dtype=ti.f32, shape=coords.shape[:-1])
field.from_numpy(coords)

print(field)