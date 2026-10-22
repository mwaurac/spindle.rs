import pytest

import spindle


def test_zeros():
    t = spindle.zeros([2, 2], dtype=spindle.bf16)
    assert t.shape == [2, 2]
    assert t.dtype == spindle.bf16
    assert t.device == spindle.cpu
    assert t.numel() == 4
    assert repr(t) == "tensor(shape=[2, 2], dtype=bf16, device=cpu)"
