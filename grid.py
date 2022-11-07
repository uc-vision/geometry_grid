
def grid_3d(dim:Array3i) -> AABox:
  x, y, z = dr.meshgrid(
      dr.arange(Float, dim.x),
      dr.arange(Float, dim.y),
      dr.arange(Float, dim.z)
  )