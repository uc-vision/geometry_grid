import taichi as ti

def placed_field(snode, dtype):
  field = ti.field(dtype)
  snode.place(field)
  return field


def block_bitmask(size, chunk):
    cell_blocks:ti.SNode = ti.root.bitmasked(ti.ijk, [1 + x//chunk for x in size])
    return cell_blocks.bitmasked(ti.ijk, (chunk,chunk,chunk))
