import taichi as ti

@ti.dataclass
class FloatIndex:
  i : ti.u32
  f : ti.f32

@ti.func
def pack_index(pair:FloatIndex) -> ti.u64:
  f = ti.bit_cast(pair.f, ti.i32)
  return  (ti.cast(pair.i, ti.u64) << 32) | ti.cast(f, ti.u64)

@ti.func
def unpack_index(p:ti.u64) -> FloatIndex:
  f = ti.cast(p, ti.u32)
  return FloatIndex(
    ti.cast(p >> 32, ti.u32), 
    ti.bit_cast(f, ti.f32))


