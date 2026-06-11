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


BITWISE_SHAPES = (
    ((512, 1024), (512, 1024)),
    ((256, 512), (1, 512)),
    ((256, 512), (256, 1)),
    ((1, 512), (256, 512)),
    ((256, 1), (256, 512)),
    ((1024,), ()),
    ((), (1024,)),
)


def broadcast_tensors(a, b):
    shape = torch.broadcast_shapes(a.shape, b.shape)
    a = a.expand(shape).contiguous()
    b = b.expand(shape).contiguous()
    return a, b

def NPU_bitwise_left_shift(res_a, res_b):
    res_a, res_b = broadcast_tensors(res_a, res_b)
    return res_a << res_b  # torch.bitwise_left_shift(res_a, res_b)
    # [W610 18:26:03.055846726 VariableFallbackKernel.cpp:250] Warning: CAUTION: The operator 'aten::bitwise_left_shift.Tensor_out' is not currently supported on the NPU backend and will fall back to run on the CPU. This may have performance implications. (function npu_cpu_fallback)
    # И кто эту шляпу придумал (недодумал)?! Зато оператор '<<' ещё как не fallback'ается.
    # Ещё и внутри op-plugin накосячили, что требуется теперь broadcast_tensors из-за недопустимного прямого expand внутри -_-

@pytest.mark.bitwise_left_shift
@pytest.mark.parametrize("shapes", BITWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_INT_DTYPES + (torch.uint8,))
def test_accuracy_bitwise_left_shift(shapes, dtype):
    shape_a, shape_b = shapes
    res_a = torch.randint(0, 100, shape_a, dtype=dtype, device=device)
    res_b = torch.randint(0, 8, shape_b, dtype=dtype, device=device)
    ref_a = to_reference(res_a)
    ref_b = to_reference(res_b)

    ref_out = torch.bitwise_left_shift(ref_a, ref_b)
    res_out = NPU_bitwise_left_shift(res_a, res_b)
    assert_close(res_out, ref_out, dtype)


def NPU_bitwise_right_shift(res_a, res_b):
    res_a, res_b = broadcast_tensors(res_a, res_b)
    return res_a >> res_b

@pytest.mark.bitwise_right_shift
@pytest.mark.parametrize("shapes", BITWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_INT_DTYPES + (torch.uint8,))
def test_accuracy_bitwise_right_shift(shapes, dtype):
    shape_a, shape_b = shapes
    res_a = torch.randint(0, 100, shape_a, dtype=dtype, device=device)
    res_b = torch.randint(0, 8, shape_b, dtype=dtype, device=device)
    ref_a = to_reference(res_a)
    ref_b = to_reference(res_b)

    ref_out = torch.bitwise_right_shift(ref_a, ref_b)
    res_out = NPU_bitwise_right_shift(res_a, res_b)
    assert_close(res_out, ref_out, dtype)


INPLACE_BITWISE_SHAPES = [
    ((512, 1024), (512, 1024)),
    ((256, 512), (1, 512)),
    ((256, 512), (256, 1)),
    ((1024,), ()),
    # а здесь убрали те 3 формы, которые ломаются из-за op-plugin -_-
]


@pytest.mark.bitwise_left_shift
@pytest.mark.parametrize("shapes", INPLACE_BITWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_INT_DTYPES + (torch.uint8,))
def test_accuracy_bitwise_left_shift_(shapes, dtype):
    shape_a, shape_b = shapes
    res_a = torch.randint(0, 100, shape_a, dtype=dtype, device=device)
    res_b = torch.randint(0, 8, shape_b, dtype=dtype, device=device)
    ref_a = to_reference(res_a.clone())
    ref_b = to_reference(res_b)

    ref_a.bitwise_left_shift_(ref_b)
    res_a = res_a << res_b  # res_a.bitwise_left_shift_(res_b)
    assert_close(res_a, ref_a, dtype)


@pytest.mark.bitwise_right_shift
@pytest.mark.parametrize("shapes", INPLACE_BITWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_INT_DTYPES + (torch.uint8,))
def test_accuracy_bitwise_right_shift_(shapes, dtype):
    shape_a, shape_b = shapes
    res_a = torch.randint(0, 100, shape_a, dtype=dtype, device=device)
    res_b = torch.randint(0, 8, shape_b, dtype=dtype, device=device)
    ref_a = to_reference(res_a.clone())
    ref_b = to_reference(res_b)

    ref_a.bitwise_right_shift_(ref_b)
    res_a = res_a >> res_b  # res_a.bitwise_right_shift_(res_b)
    assert_close(res_a, ref_a, dtype)


@pytest.mark.bitwise_not
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwisenot(shape, dtype):
    if dtype in BOOL_TYPES:
        inp = torch.randint(0, 2, size=shape, dtype=dtype, device=device)
    else:
        inp = torch.randint(-0x7FFF, 0x7FFF, size=shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp)

    ref_out = torch.bitwise_not(ref_inp)
    res_out = torch.bitwise_not(inp)

    assert_equal(res_out, ref_out)


@pytest.mark.inplace
@pytest.mark.bitwise_not_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwisenot_(shape, dtype):
    if dtype in BOOL_TYPES:
        res_inp = torch.randint(0, 2, size=shape, dtype=dtype, device=device)
    else:
        res_inp = torch.randint(-0x7FFF, 0x7FFF, size=shape, dtype=dtype, device=device)
    ref_inp = to_reference(res_inp.clone())

    ref_out = ref_inp.bitwise_not_()  # NOTE: there is no torch.bitwse_not_
    res_out = res_inp.bitwise_not_()

    assert_equal(res_out, ref_out)


@pytest.mark.cos
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_cos(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.cos(ref_inp)
    res_out = torch.cos(inp)

    assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.cos_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_cos_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp.clone(), True)

    ref_out = torch.cos_(ref_inp)
    res_out = torch.cos_(inp)

    assert_close(res_out, ref_out, dtype)


@pytest.mark.exp
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_exp(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.exp(ref_inp)
    res_out = torch.exp(inp)

    assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.exp_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_exp_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp.clone(), True)

    ref_out = torch.exp_(ref_inp)
    res_out = torch.exp_(inp)

    assert_close(res_out, ref_out, dtype)


@pytest.mark.exp
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_exp_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.exp(ref_inp, out=ref_out)
    res_out = torch.empty_like(inp)
    torch.exp(inp, out=res_out)

    assert_close(res_out, ref_out, dtype)


@pytest.mark.exp2
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_exp2(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.exp2(ref_inp)
    res_out = torch.exp2(inp)

    assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.exp2_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_exp2_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp.clone(), True)

    ref_out = torch.exp2_(ref_inp)
    res_out = torch.exp2_(inp)

    assert_close(res_out, ref_out, dtype)
