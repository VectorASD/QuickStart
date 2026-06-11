# Адаптировано на основе публичного кода из репозитория FlagGems:
# https://github.com/flagos-ai/FlagGems/blob/master/tests/accuracy_utils.py
# Используются только общепринятые практики и сигнатуры хелперов,
# которые не являются объектами авторского права или секретом.

import torch
import numpy as np

from .custom_functional import *  # патчим torch новыми функциями

import random
import time
import os



import warnings
from _pytest.warning_types import PytestUnknownMarkWarning

warnings.filterwarnings("ignore", category=PytestUnknownMarkWarning)

device = "npu"
os.environ["NOT_NPU_QUIET"] = '1'



# примитивные типы

bf16_is_supported = True
fp64_is_supported = True
int64_is_supported = True

PRIMARY_FLOAT_DTYPES = (torch.float16, torch.float32)
FLOAT_DTYPES = (*PRIMARY_FLOAT_DTYPES, *((torch.bfloat16,) if bf16_is_supported else ()))
ALL_FLOAT_DTYPES = (*FLOAT_DTYPES, *((torch.float64,) if fp64_is_supported else ()))

INT_DTYPES = (torch.int16, torch.int32)
ALL_INT_DTYPES = (*INT_DTYPES, *((torch.int64,) if int64_is_supported else ()))

BOOL_TYPES = (torch.bool,)

COMPLEX_DTYPES = (torch.complex32, torch.complex64)



# формы

QUICK_MODE = False

DISTRIBUTION_SHAPES = ((20, 320, 15),)
POINTWISE_SHAPES = (((2, 19, 7),) if QUICK_MODE else ((), (1,), (1024, 1024), (20, 320, 15), (16, 128, 64, 60), (16, 7, 57, 32, 29)))

SWIGLU_SPECIAL_SHAPES = (((2, 19, 8),) if QUICK_MODE else (
    (2,),
    (64,),
    (32, 64),
    (256, 512),
    (1, 128),
    (8, 16, 32),
    (16, 32, 64),
    (20, 320, 16),
    (4, 8, 16, 32),
    (8, 16, 32, 64),
    (10,),
    (20, 30),
))



# функционал

RESOLUTION = {
    torch.bool: 0,
    torch.uint8: 0,
    torch.int8: 0,
    torch.int16: 0,
    torch.int32: 0,
    torch.int64: 0,
    torch.float8_e4m3fn: 1e-3,
    torch.float8_e5m2: 1e-3,
    torch.float8_e4m3fnuz: 1e-3,
    torch.float8_e5m2fnuz: 1e-3,
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.bfloat16: 0.016,
    torch.float64: 1e-7,
    torch.complex32: 1e-3,
    torch.complex64: 1.3e-6,
}

def unsqueeze_tuple(t, max_len):
    return t + (1,) * (max_len - len(t))

def unsqueeze_tensor(inp, max_ndim):
    for _ in range(inp.ndim, max_ndim):
        inp = inp.unsqueeze(-1)
    return inp

def assert_close(res, ref, dtype, equal_nan=False, reduce_dim=1, atol=1e-4):
    if dtype is None:
        dtype = torch.float32
    assert res.dtype == dtype
    ref = ref.to(dtype=dtype, device="cpu")
    res = res.to(device="cpu")
    rtol = RESOLUTION[dtype]
    torch.testing.assert_close(res, ref, atol=atol * reduce_dim, rtol=rtol, equal_nan=equal_nan)

def assert_equal(res, ref, equal_nan=False):
    res, ref = res.cpu(), ref.cpu()
    torch.testing.assert_close(res, ref, atol=0, rtol=0, equal_nan=equal_nan)

def to_reference(inp, upcast=False):
    if inp is None:
        return None
    ref_inp = inp.to("cpu")
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp

def to_cpu(res, ref):
    if isinstance(res, torch.Tensor) and isinstance(ref, torch.Tensor):
        res = res.to("cpu")
        assert ref.device == torch.device("cpu")
    return res

def init_seed(seed = None):
    if seed is None:
        seed = time.time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
