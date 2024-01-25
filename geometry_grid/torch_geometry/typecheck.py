from beartype import beartype
import jaxtyping
from jaxtyping import jaxtyped, Float32, Int32, Bool
from torch import Tensor

from tensorclass import TensorClass


NInt32 = Int32[Tensor, "N"]
NFloat32 = Float32[Tensor, "N"]


NVec1 = Float32[Tensor, "N 1"]
NVec2 = Float32[Tensor, "N 2"]
NVec3 = Float32[Tensor, "N 3"]
NVec4 = Float32[Tensor, "N 4"]

Vec1 = Float32[Tensor, "1"]
Vec2 = Float32[Tensor, "2"]
Vec3 = Float32[Tensor, "3"]
Vec4 = Float32[Tensor, "4"]

IVec1 = Int32[Tensor, "N 1"]
IVec2 = Int32[Tensor, "N 2"]
IVec3 = Int32[Tensor, "N 3"]
IVec4 = Int32[Tensor, "N 4"]


def typechecked(f):
    return beartype(jaxtyped(f))
