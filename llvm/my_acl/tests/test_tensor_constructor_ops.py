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

# sudo python -m pip install --upgrade pip
# sudo pip install sqlalchemy pybind11 pytest pytest-repeat pytest-xdist setuptools numpy pyyaml decorator einops scipy attrs psutil pandas transformers
# pytest-repeat: --count=100
# pytest-xdist: -n=4    or -n=auto (физические ядра)   or -n=logical (логические ядра)

import pytest
import torch

from .accuracy_utils import (
    ALL_FLOAT_DTYPES,
    ALL_INT_DTYPES,
    BOOL_TYPES,
    DISTRIBUTION_SHAPES,
    FLOAT_DTYPES,
    POINTWISE_SHAPES,
    assert_equal,
    to_reference,
    device,
)

# pytest test_tensor_constructor_ops.py -m rand -sv
# pytest test_tensor_constructor_ops.py -m rand --count-100



@pytest.mark.rand
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rand(shape, dtype):
    res_out = torch.rand(shape, dtype=dtype, device=device)
    ref_out = to_reference(res_out)

    assert (ref_out <= 1.0).all()
    assert (ref_out >= 0.0).all()


@pytest.mark.randn
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_randn(shape, dtype):
    res_out = torch.randn(shape, dtype=dtype, device=device)
    ref_out = to_reference(res_out)

    mean = torch.mean(ref_out)
    std = torch.std(ref_out)
    assert torch.abs(mean) < 0.01
    assert torch.abs(std - 1) < 0.01


@pytest.mark.rand_like
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rand_like(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device=device)
    res_out = torch.rand_like(x)  # randn to rand :)
    ref_out = to_reference(res_out)

    assert (ref_out <= 1.0).all()
    assert (ref_out >= 0.0).all()


@pytest.mark.randn_like
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_randn_like(shape, dtype):
    x = torch.rand(size=shape, dtype=dtype, device=device)

    res_out = torch.randn_like(x)  # rand to randn :)
    ref_out = to_reference(res_out)

    mean = torch.mean(ref_out)
    std = torch.std(ref_out)
    assert torch.abs(mean) < 0.01
    assert torch.abs(std - 1) < 0.01


@pytest.mark.zeros
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", BOOL_TYPES + ALL_INT_DTYPES + ALL_FLOAT_DTYPES)
def test_accuracy_zeros(shape, dtype):
    res_out = torch.zeros(shape, device=device)
    assert_equal(res_out, torch.zeros(shape))

    res_out = torch.zeros(shape, dtype=dtype, device=device)
    assert_equal(res_out, torch.zeros(shape, dtype=dtype))


@pytest.mark.zero_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", BOOL_TYPES + ALL_INT_DTYPES + ALL_FLOAT_DTYPES)
def test_accuracy_zero_(shape, dtype):
    res_out = torch.ones(shape, dtype=dtype, device=device)
    ref_out = to_reference(res_out)

    ref_out.zero_()
    res_out.zero_()

    assert_equal(res_out, ref_out)


@pytest.mark.ones
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", BOOL_TYPES + ALL_INT_DTYPES + ALL_FLOAT_DTYPES)
def test_accuracy_ones(shape, dtype):
    # without dtype
    res_out = torch.ones(shape, device=device)
    assert_equal(res_out, torch.ones(shape))

    # with dtype
    res_out = torch.ones(shape, dtype=dtype, device=device)
    assert_equal(res_out, torch.ones(shape, dtype=dtype))


@pytest.mark.full
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", BOOL_TYPES + ALL_INT_DTYPES + ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("fill_value", (3.1415926, 2, False))
def test_accuracy_full(shape, dtype, fill_value):
    # without dtype
    ref_out = torch.full(shape, fill_value)
    res_out = torch.full(shape, fill_value, device=device)

    assert_equal(res_out, ref_out)

    # with dtype
    ref_out = torch.full(shape, fill_value, dtype=dtype)
    res_out = torch.full(shape, fill_value, dtype=dtype, device=device)

    assert_equal(res_out, ref_out)


@pytest.mark.zeros_like
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_zeros_like(shape, dtype):
    inp = torch.empty(size=shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp)

    ref_out = torch.zeros_like(ref_inp)
    res_out = torch.zeros_like(inp)

    assert_equal(res_out, ref_out)


@pytest.mark.ones_like
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_ones_like(shape, dtype):
    inp = torch.empty(size=shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp)

    ref_out = torch.ones_like(ref_inp)
    res_out = torch.ones_like(inp)

    assert_equal(res_out, ref_out)


@pytest.mark.full_like
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", BOOL_TYPES + ALL_INT_DTYPES + ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("fill_value", (3.1415926, 2, False))
def test_accuracy_full_like(shape, dtype, fill_value):
    inp = torch.empty(size=shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp)

    # without dtype
    ref_out = torch.full_like(ref_inp, fill_value)
    res_out = torch.full_like(inp, fill_value)
    assert_equal(res_out, ref_out, equal_nan=True)

    # with dtype
    ref_out = torch.full_like(ref_inp, fill_value, dtype=dtype)
    res_out = torch.full_like(inp, fill_value, dtype=dtype)
    assert_equal(res_out, ref_out, equal_nan=True)


@pytest.mark.randperm
@pytest.mark.parametrize("n", (123, 12345, 123456))
@pytest.mark.parametrize("dtype", ALL_INT_DTYPES)
def test_accuracy_randperm(n, dtype):
    if n > torch.iinfo(dtype).max:
        pytest.skip(f"n > {hex(torch.iinfo(dtype).max)}")

    ref_out = torch.randperm(n, dtype=dtype)
    res_out = torch.randperm(n, dtype=dtype, device=device)

    sorted_ref, _ = torch.sort(ref_out)
    sorted_res, _ = torch.sort(res_out)
    assert_equal(sorted_res, sorted_ref)


@pytest.mark.eye
@pytest.mark.parametrize(
    "shape",
    (
        (256, 1024),
        (1024, 256),
        (8192, 4096),
        (4096, 8192),
        *((2**d, 2**d) for d in range(7, 13)),
    ),
)
@pytest.mark.parametrize("dtype", ALL_INT_DTYPES + ALL_FLOAT_DTYPES + BOOL_TYPES)
def test_accuracy_eye(shape, dtype):
    n, m = shape

    # test eye(n, m) without dtype
    ref_out = torch.eye(n, m)
    res_out = torch.eye(n, m, device=device)
    assert_equal(res_out, ref_out)

    # with dtype
    ref_out = torch.eye(n, m, dtype=dtype)
    res_out = torch.eye(n, m, dtype=dtype, device=device)
    assert_equal(res_out, ref_out)

    # test eye(n)
    ref_out = torch.eye(n)
    res_out = torch.eye(n, device=device)
    assert_equal(res_out, ref_out)

    # with dtype
    ref_out = torch.eye(n, dtype=dtype)
    res_out = torch.eye(n, dtype=dtype, device=device)
    assert_equal(res_out, ref_out)


@pytest.mark.one_hot
def test_accuracy_one_hot():
    one_hot = torch.nn.functional.one_hot

    x = torch.tensor((3, 4, 1, 0), device=device, dtype=torch.int64)
    t = one_hot(x)
    expected = torch.tensor((
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
        (0, 1, 0, 0, 0),
        (1, 0, 0, 0, 0)
    ))
    assert_equal(t, expected)

    t = one_hot(x, -1)
    expected = torch.tensor((
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
        (0, 1, 0, 0, 0),
        (1, 0, 0, 0, 0)
    ))
    assert_equal(t, expected)

    t = one_hot(x, 6)
    expected = torch.tensor((
        (0, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 1, 0),
        (0, 1, 0, 0, 0, 0),
        (1, 0, 0, 0, 0, 0),
    ))
    assert_equal(t, expected)

    x2 = torch.tensor(((3, 4), (1, 0)), device=device, dtype=torch.int64)
    t = one_hot(x2)
    expected = torch.tensor((
        ((0, 0, 0, 1, 0),
         (0, 0, 0, 0, 1)),
        ((0, 1, 0, 0, 0),
         (1, 0, 0, 0, 0))
    ))
    assert_equal(t, expected)

    x0 = torch.tensor(4, device=device, dtype=torch.int64)
    t = one_hot(x0)
    expected = torch.tensor((0, 0, 0, 0, 1))
    assert_equal(t, expected)

    x_empty = torch.empty((4, 0), dtype=torch.long, device=device)
    t = one_hot(x_empty, 100)
    expected = torch.empty((4, 0, 100), dtype=torch.long)
    assert_equal(t, expected)

    with pytest.raises(RuntimeError):
        one_hot(torch.empty((4, 0), dtype=torch.long, device=device))

    with pytest.raises(RuntimeError):
        one_hot(torch.tensor((3, 4, 1, 0), dtype=torch.long, device=device), -2)
