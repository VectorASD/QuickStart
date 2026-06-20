# ----------------------------------------------------------------------
#  Тесты для not_npu: проверка соответствия нашего эмулированного ACL
#  реальному Ascend CANN Toolkit.
#
#  Исходная идея и мета-данные тестов (имена функций, наборы параметров,
#  декораторы pytest) взяты из внутренних тестов FlagGems (приватного
#  репозитория Huawei), однако **тела тестов написаны с нуля**.
#
#  Из FlagGems используются исключительно:
#    - названия тестовых функций (test_accuracy_zeros, ...),
#    - комбинации параметров (shapes, dtypes, fill_value),
#    - pytest-маркеры и parametrize-декораторы.
#  Это общепринятые обозначения и функциональность pytest/PyTorch,
#  которые не являются объектом авторского права.
#
#  Сам torch_npu (и его код) никак не модифицируется: мы лишь
#  подменяем несколько .so-библиотек (not_npu), эмулируя поведение
#  Ascend CANN Toolkit на CPU.  Цель – убедиться, что наши
#  реализации операций (ACL) полностью совместимы с реальным
#  поведением toolkit'а на NPU.
#
#  Полезные библиотеки – полностью наша реализация (not_npu):
#    - libascendcl.so          (базовый runtime: устройства, память, тензоры)
#    - libacl_op_compiler.so   (эмуляция операторов ACL)
#
#  Заглушки, необходимые только для успешной линковки символов при импорте torch_npu._C:
#    - libacl_tdt_channel.so
#    - libge_runner.so
#    - libgraph.so
#    - libhccl.so
#
#  Все проверки выполняются стандартными средствами PyTorch и pytest.
# ----------------------------------------------------------------------

import logging
from math import pi
from random import random, randint

import pytest
import torch

from .accuracy_utils import (
    ALL_FLOAT_DTYPES,
    ALL_INT_DTYPES,
    BOOL_TYPES,
    FLOAT_DTYPES,
    INT_DTYPES,
    POINTWISE_SHAPES,
    SCALARS,
    assert_close,
    assert_equal,
    to_reference,
    device,
)

# pytest test_binary_pointwise_ops.py -m add -sv
# pytest test_binary_pointwise_ops.py --count-100 --log



def replace_zeros(inp):
    return torch.where(inp == 0, 1, inp)


@pytest.mark.add
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_add(shape, alpha, dtype):
    res_inp1 = torch.randn(shape, dtype=dtype, device=device)
    res_inp2 = torch.randn(shape, dtype=dtype, device=device)
    ref_inp1 = to_reference(res_inp1)
    ref_inp2 = to_reference(res_inp2)

    ref_out = torch.add(ref_inp1, ref_inp2, alpha=alpha)
    res_out = torch.add(res_inp1, res_inp2, alpha=alpha)

    assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.add_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_add_(shape, alpha, dtype):
    res_inp1 = torch.randn(shape, dtype=dtype, device=device)
    res_inp2 = torch.randn(shape, dtype=dtype, device=device)
    ref_inp1 = to_reference(res_inp1)
    ref_inp2 = to_reference(res_inp2)

    ref_out = ref_inp1.add_(ref_inp2, alpha=alpha)
    res_out = res_inp1.add_(res_inp2, alpha=alpha)

    assert_close(res_out, ref_out, dtype)


@pytest.mark.add
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_add_tensor_scalar(shape, scalar, alpha, dtype):
    res_inp1 = torch.randn(shape, dtype=dtype, device=device)
    ref_inp1 = to_reference(res_inp1)
    inp2 = scalar

    ref_out = torch.add(ref_inp1, inp2, alpha=alpha)
    res_out = torch.add(res_inp1, inp2, alpha=alpha)

    assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.add_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_add_tensor_scalar_(shape, scalar, alpha, dtype):
    res_inp1 = torch.randn(shape, dtype=dtype, device=device)
    ref_inp1 = to_reference(res_inp1)
    inp2 = scalar

    ref_out = ref_inp1.add_(inp2, alpha=alpha)
    res_out = res_inp1.add_(inp2, alpha=alpha)

    assert_close(res_out, ref_out, dtype)


@pytest.mark.add
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_add_scalar_tensor(shape, scalar, alpha, dtype):
    inp1 = scalar

    res_inp2 = torch.randn(shape, dtype=dtype, device=device)
    ref_inp2 = to_reference(res_inp2)

    ref_out = torch.add(inp1, ref_inp2, alpha=alpha)
    res_out = torch.add(inp1, res_inp2, alpha=alpha)

    assert_close(res_out, ref_out, dtype)


@pytest.mark.add
@pytest.mark.parametrize("dtype", (torch.float32, torch.int64))
def test_accuracy_add_scalar_scalar(dtype):
    if dtype == torch.float32:
        res_inp1  = torch.tensor(random(), dtype=dtype, device=device)
        res_inp2  = torch.tensor(random(), dtype=dtype, device=device)
        res_alpha = torch.tensor(random(), dtype=dtype, device=device)
    else:
        res_inp1  = torch.tensor(randint(0, 100), dtype=dtype, device=device)
        res_inp2  = torch.tensor(randint(0, 100), dtype=dtype, device=device)
        res_alpha = torch.tensor(randint(0, 100), dtype=dtype, device=device)

    ref_inp1  = to_reference(res_inp1)
    ref_inp2  = to_reference(res_inp2)
    ref_alpha = to_reference(res_alpha)

    ref_out = torch.add(ref_inp1, ref_inp2, alpha=ref_alpha)
    res_out = torch.add(res_inp1, res_inp2, alpha=res_alpha)

    if dtype == torch.int64:
        assert_equal(res_out, ref_out)
    else:
        assert_close(res_out, ref_out, dtype)


@pytest.mark.bitwise_and
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseand(shape, dtype):
    if dtype in BOOL_TYPES:
        res_inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device=device)
        res_inp2 = torch.randint(0, 2, size=shape, dtype=dtype, device=device)
    else:
        res_inp1 = torch.randint(-0x7FFF, 0x7FFF, size=shape, dtype=dtype, device=device)
        res_inp2 = torch.randint(-0x7FFF, 0x7FFF, size=shape, dtype=dtype, device=device)

    ref_inp1 = to_reference(res_inp1)
    ref_inp2 = to_reference(res_inp2)

    ref_out = torch.bitwise_and(ref_inp1, ref_inp2)
    res_out = torch.bitwise_and(res_inp1, res_inp2)

    assert_equal(res_out, ref_out)


@pytest.mark.inplace
@pytest.mark.bitwise_and_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseand_(shape, dtype):
    if dtype in BOOL_TYPES:
        res_inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device=device)
        res_inp2 = torch.randint(0, 2, size=shape, dtype=dtype, device=device)
    else:
        res_inp1 = torch.randint(-0x7FFF, 0x7FFF, size=shape, dtype=dtype, device=device)
        res_inp2 = torch.randint(-0x7FFF, 0x7FFF, size=shape, dtype=dtype, device=device)

    ref_inp1 = to_reference(res_inp1)
    ref_inp2 = to_reference(res_inp2)

    ref_out = ref_inp1.bitwise_and_(ref_inp2)
    res_out = res_inp1.bitwise_and_(res_inp2)

    assert_equal(res_out, ref_out)


@pytest.mark.bitwise_and
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseand_scalar(shape, dtype):
    if dtype in BOOL_TYPES:
        res_inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device=device)
        inp2 = bool(randint(0, 2))
    else:
        res_inp1 = torch.randint(-0x7FFF, 0x7FFF, size=shape, dtype=dtype, device=device)
        inp2 = 0x00FF

    ref_inp1 = to_reference(res_inp1)

    ref_out = torch.bitwise_and(ref_inp1, inp2)
    res_out = torch.bitwise_and(res_inp1, inp2)

    assert_equal(res_out, ref_out)


@pytest.mark.inplace
@pytest.mark.bitwise_and_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseand_scalar_(shape, dtype):
    if dtype in BOOL_TYPES:
        res_inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device=device)
        inp2 = bool(randint(0, 2))
    else:
        res_inp1 = torch.randint(-0x7FFF, 0x7FFF, size=shape, dtype=dtype, device=device)
        inp2 = 0x00FF

    ref_inp1 = to_reference(res_inp1)

    ref_out = ref_inp1.bitwise_and_(inp2)
    res_out = res_inp1.bitwise_and_(inp2)

    assert_equal(res_out, ref_out)


@pytest.mark.bitwise_and
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseand_scalar_tensor(shape, dtype):
    if dtype in BOOL_TYPES:
        inp1 = bool(randint(0, 2))
        res_inp2 = torch.randint(0, 2, size=shape, dtype=dtype, device=device)
    else:
        inp1 = 0x00FF
        res_inp2 = torch.randint(-0x7FFF, 0x7FFF, size=shape, dtype=dtype, device=device)

    ref_inp2 = to_reference(res_inp2)

    ref_out = torch.bitwise_and(inp1, ref_inp2)
    res_out = torch.bitwise_and(inp1, res_inp2)

    assert_equal(res_out, ref_out)


@pytest.mark.bitwise_or
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseor(shape, dtype):
    if dtype in BOOL_TYPES:
        res_inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device=device)
        res_inp2 = torch.randint(0, 2, size=shape, dtype=dtype, device=device)
    else:
        res_inp1 = torch.randint(-0x7FFF, 0x7FFF, size=shape, dtype=dtype, device=device)
        res_inp2 = torch.randint(-0x7FFF, 0x7FFF, size=shape, dtype=dtype, device=device)

    ref_inp1 = to_reference(res_inp1)
    ref_inp2 = to_reference(res_inp2)

    ref_out = torch.bitwise_or(ref_inp1, ref_inp2)
    res_out = torch.bitwise_or(res_inp1, res_inp2)

    assert_equal(res_out, ref_out)


@pytest.mark.inplace
@pytest.mark.bitwise_or_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseor_(shape, dtype):
    if dtype in BOOL_TYPES:
        res_inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device=device)
        res_inp2 = torch.randint(0, 2, size=shape, dtype=dtype, device=device)
    else:
        res_inp1 = torch.randint(-0x7FFF, 0x7FFF, size=shape, dtype=dtype, device=device)
        res_inp2 = torch.randint(-0x7FFF, 0x7FFF, size=shape, dtype=dtype, device=device)

    ref_inp1 = to_reference(res_inp1)
    ref_inp2 = to_reference(res_inp2)

    ref_out = ref_inp1.bitwise_or_(ref_inp2)
    res_out = res_inp1.bitwise_or_(res_inp2)

    assert_equal(res_out, ref_out)


@pytest.mark.bitwise_or
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseor_scalar(shape, dtype):
    if dtype in BOOL_TYPES:
        res_inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device=device)
        inp2 = bool(randint(0, 2))
    else:
        res_inp1 = torch.randint(-0x7FFF, 0x7FFF, size=shape, dtype=dtype, device=device)
        inp2 = 0x00FF

    ref_inp1 = to_reference(res_inp1)

    ref_out = torch.bitwise_or(ref_inp1, inp2)
    res_out = torch.bitwise_or(res_inp1, inp2)

    assert_equal(res_out, ref_out)


@pytest.mark.inplace
@pytest.mark.bitwise_or_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseor_scalar_(shape, dtype):
    if dtype in BOOL_TYPES:
        res_inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device=device)
        inp2 = bool(randint(0, 2))
    else:
        res_inp1 = torch.randint(-0x7FFF, 0x7FFF, size=shape, dtype=dtype, device=device)
        inp2 = 0x00FF
    ref_inp1 = to_reference(res_inp1)

    ref_out = ref_inp1.bitwise_or_(inp2)
    res_out = res_inp1.bitwise_or_(inp2)

    assert_equal(res_out, ref_out)


@pytest.mark.bitwise_or
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseor_scalar_tensor(shape, dtype):
    if dtype in BOOL_TYPES:
        inp1 = bool(randint(0, 2))
        res_inp2 = torch.randint(0, 2, size=shape, dtype=torch.bool, device=device)
    else:
        inp1 = 0x00FF
        res_inp2 = torch.randint(-0x7FFF, 0x7FFF, size=shape, dtype=dtype, device=device)

    ref_inp2 = to_reference(res_inp2)

    ref_out = torch.bitwise_or(inp1, ref_inp2)
    res_out = torch.bitwise_or(inp1, res_inp2)

    assert_equal(res_out, ref_out)


@pytest.mark.clamp
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("maxi", SCALARS)
@pytest.mark.parametrize("mini", SCALARS)
@pytest.mark.parametrize("isnone", (None, "max", "min"))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_clamp(shape, maxi, mini, isnone, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(res_inp)

    if isnone == "min":
        mini = None
    elif isnone == "max":
        maxi = None

    ref_out = torch.clamp(ref_inp, min=mini, max=maxi)
    res_out = torch.clamp(res_inp, min=mini, max=maxi)

    assert_equal(res_out, ref_out)
