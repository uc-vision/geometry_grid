import taichi as ti

@ti.func
def pack_index(index:ti.i32, value:ti.f32) -> ti.u64:
  f = ti.bit_cast(value, ti.u32)
  i = ti.bit_cast(index, ti.u32)
  return (ti.cast(i, ti.u64)) | (ti.cast(f, ti.u64)  << 32)



@ti.func
def unpack_index(p:ti.u64):
  f = ti.cast(p  >> 32, ti.u32)
  return (
    ti.cast(p, ti.i32), 
    ti.bit_cast(f, ti.f32))


