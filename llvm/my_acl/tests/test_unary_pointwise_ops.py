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

import pytest
import torch

TE_AVAILABLE = True

from .accuracy_utils import (
    ALL_FLOAT_DTYPES,
    ALL_INT_DTYPES,
    BOOL_TYPES,
    COMPLEX_DTYPES,
    FLOAT_DTYPES,
    INT_DTYPES,
    POINTWISE_SHAPES,
    SWIGLU_SPECIAL_SHAPES,
    assert_close,
    assert_equal,
    to_reference,
    unsqueeze_tensor,
    unsqueeze_tuple,
    device,
)

# pytest test_unary_pointwise_ops.py -m abs -sv
# pytest test_unary_pointwise_ops.py --count-100 --log



@pytest.mark.abs
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_abs(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp)

    ref_out = torch.abs(ref_inp)
    res_out = torch.abs(inp)

    assert_equal(res_out, ref_out)


@pytest.mark.inplace
@pytest.mark.abs_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_abs_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp.clone())

    ref_out = torch.abs_(ref_inp)
    res_out = torch.abs_(inp)

    assert_equal(res_out, ref_out)


@pytest.mark.acos
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_acos(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp)

    ref_out = torch.acos(ref_inp)
    res_out = torch.acos(inp)

    assert_close(res_out, ref_out, dtype, True)


@pytest.mark.angle
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", COMPLEX_DTYPES + FLOAT_DTYPES + ALL_INT_DTYPES + BOOL_TYPES)
def test_accuracy_angle(shape, dtype):
    if dtype in BOOL_TYPES:
        inp = torch.randint(0, 2, size=shape, dtype=dtype, device=device)
    elif dtype in ALL_INT_DTYPES:
        inp = torch.randint(-0x7FFF, 0x7FFF, size=shape, dtype=dtype, device=device)
    elif dtype in COMPLEX_DTYPES + FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device="cpu").to(device)
        # какие-то проблемы в самом torch_npu - выделяет меньше памяти под тензор, чем надо?
        # это приводит к Segmentation fault при вызове to_reference(inp)
    ref_inp = to_reference(inp)
    try:
        ref_out = torch.angle(ref_inp)
    except RuntimeError as e:
        if "angle_cpu" in str(e) and "ComplexHalf" in str(e):
            pytest.skip("Skipping angle ComplexHalf for unsupported dtype on CPU")
        else:
            raise
    ref_out = torch.angle(ref_inp)
    res_out = torch.angle(inp)
    dtype_out = res_out.dtype
    assert_close(res_out, ref_out, dtype_out)
