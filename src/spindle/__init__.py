from typing import List

from spindle._core import Tensor

f16 = "f16"
bf16 = "bf16"
f32 = "f32"
f64 = "f64"
i32 = "i32"
i64 = "i64"
q8_0 = "q8_0"
q4_0 = "q4_0"
q4_k = "q4_k"
q6_k = "q6_k"

cpu = "cpu"


def zeros(shape: List[float], dtype: str = f32, device: str = cpu) -> Tensor:
    return Tensor.zeros(shape, dtype, device)


__all__ = [
    "Tensor",
    "bf16",
    "cpu",
    "f16",
    "f32",
    "f64",
    "i32",
    "i64",
    "q4_0",
    "q4_k",
    "q6_k",
    "q8_0",
]
