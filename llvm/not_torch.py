from __future__ import annotations

import torch
from triton.runtime.cache import FileCacheManager

import os
import inspect
import types, sys
from pprint import pformat

from typing import Any, TypeAlias, Union
from enum import IntEnum



os.environ["TRITON_NPU_COMPILER_PATH"] = os.path.expanduser("~")



def npu_mod():
    # {"load_kernel_binary", loadKernelBinary, METH_VARARGS, "Load NPU kernel binary into NPU driver"},
    def get_arch(*args):
        "Get soc version of NPU"
        return "Ascend950PR_957c"
    def get_aicore_num(*args):
        "Get the number of AI core"
        return 28
	# {"create_stream", createStream, METH_VARARGS, "Create a stream"},
    def read_data_from_file(filename: str, buffer: bytearray):
        "Read binary file into the array already allocated"
        print("[READ FILE]:", filename)
        with open(filename, "rb") as f:
            data = f.read()
        if len(buffer) < len(data):
            raise ValueError("Buffer too small for file contents")
        buffer[:len(data)] = data
    def write_data_to_file(filename: str, buffer: bytes, num_bytes: int):
        "Write an array to a binary file"
        print("[WRITE FILE]:", filename)
        with open(filename, "wb") as f:
            f.write(buffer[:num_bytes])
	# {"allocate_device_memory", allocateDeviceMemory, METH_VARARGS, "Allocate device memory"},
	# {"allocate_host_memory", allocateHostMemory, METH_VARARGS, "Allocate host memory"},
	# {"copy_memory", copyMemory, METH_VARARGS, "Copy data between host and device"},

src = inspect.getsource(npu_mod)
npu_utils_src = src.replace('def npu_mod():', 'if 1:')

real_get_file = FileCacheManager.get_file
def get_file(self, filename):
    if filename != 'npu_utils.so':
        return real_get_file(self, filename)
    cache = FileCacheManager("abcd")
    cache.put(npu_utils_src, "npu_utils.py", binary=False)
    path = cache.get_file("npu_utils.py")
    if path is None:
        path = cache.put(npu_utils_src, "npu_utils.py", binary=False)
    return path
FileCacheManager.get_file = get_file



dummy     = types.ModuleType("torch_npu")
dummy_C   = types.ModuleType("torch_npu._C")
dummy_npu = types.ModuleType("torch_npu.npu")
dummy._C  = dummy_C
dummy.npu = dummy_npu
dummy.__file__     = "dummy"
dummy_C.__file__   = "dummy_C"
dummy_npu.__file__ = "dummy_npu"
dummy_C._npu_getCurrentRawStream = lambda device_id: 123 + device_id
sys.modules["torch_npu"]    = dummy
sys.modules["torch_npu._C"] = dummy_C

class npu:
    device_id = 0
    def current_device():
        return npu.device_id
    def set_device(device_id):
        npu.device_id = device_id
    def device_count():
        return 1 # используется flag_gems
    def get_device_capability(device_id):
        return (0, 0) # (9, 0) # или даже (10, 0), никто пока это не знает :)
        # мне же хуже, т.к. будет SUPPORTED_FP8_DTYPE = torch.float8_e4m3fn вместо torch.float32
        # а в реальных DTS там вообще будет (0, 0) + warning 'Failed to get device properties for device_id=0, fallback to None'
        # т.е. лучше сделать заглушку, чтобы выкинуть warning, но имитировать условия, близкие к DTS
    def is_available():
        return npu.device_count() > 0

dummy_npu.__dict__.update(npu.__dict__)
torch.npu = npu



def attr_extractor(f_name, obj, *names):
    attrs = []
    for name in names:
        try:    value = getattr(obj, name)
        except: value = ...
        attrs.append(f"{name}={'...' if value is Ellipsis else repr(value)}")
    return f"{f_name}({', '.join(attrs)})"

def dtype_extractor():
    for name in dir(torch):
        value = getattr(torch, name)
        if isinstance(value, torch.dtype):
            try: is_signed = '01'[value.is_signed]
            except RuntimeError: is_signed = None
            try: to_complex = str(value.to_complex()).split(".")[-1]
            except RuntimeError: to_complex = None
            to_real = str(value.to_real()).split(".")[-1]
            try:
                info = attr_extractor("finfo._maker", torch.finfo(value), "bits", "resolution", "min", "max", "eps", "smallest_normal", "tiny", "dtype")
            except TypeError:
                try:
                    info = attr_extractor("iinfo._maker", torch.iinfo(value), "bits", "min", "max", "dtype")
                except TypeError: info = None
            print(f"{name.ljust(10)} = dtype({name!r}, {'01'[value.is_complex]}, {'01'[value.is_floating_point]}, {is_signed}, {value.itemsize}, {to_complex!r}, {to_real!r}, {info})")
    exit()
# dtype_extractor()

def common_extractor():
    for name in dir(torch):
        value = getattr(torch, name)
        if isinstance(value, torch.memory_format):
            name = str(value).split(".")[1]
            print(f"{name} = memory_format({name!r})")
    exit()
# common_extractor()



class _finfo:
    @classmethod
    def _maker(cls, bits: int, min: float, max: float, eps: float, tiny: float, smallest_normal: float, resolution: float, dtype: str):
        self = cls()
        self.bits            = bits
        self.min             = min
        self.max             = max
        self.eps             = eps
        self.tiny            = tiny
        self.smallest_normal = smallest_normal
        self.resolution      = resolution
        self.dtype           = dtype
        return self

    # багоимитатор :)))
    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        if value is ...:
            raise TypeError("bad argument type for built-in operation")
        return value

    def __repr__(self):
        attrs = []
        for name in ("bits", "resolution", "min", "max", "eps", "smallest_normal", "tiny", "dtype"):
            value = getattr(self, name)
            attrs.append(f"{name}={'...' if value is Ellipsis else repr(value)}")
        return f"finfo({', '.join(attrs)})"

class _iinfo:
    @classmethod
    def _maker(cls, bits: int, min: float, max: float, dtype: str):
        self = cls()
        self.bits            = bits
        self.min             = min
        self.max             = max
        self.dtype           = dtype
        return self

    # багоимитатор :)))
    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        if value is ...:
            raise TypeError("bad argument type for built-in operation")
        return value

    def __repr__(self):
        attrs = []
        for name in ("bits", "min", "max", "dtype"):
            value = getattr(self, name)
            attrs.append(f"{name}={'...' if value is Ellipsis else repr(value)}")
        return f"iinfo({', '.join(attrs)})"

class NotTorch:
    class dtype:
        index = {}
        def __init__(self, name, is_complex, is_floating_point, is_signed, itemsize, complex_name, real_name, info=...):
            self.name              = name
            self.is_complex        = is_complex
            self.is_floating_point = is_floating_point
            self.is_signed         = is_signed
            self.itemsize          = itemsize
            self.complex_name      = complex_name
            self.real_name         = real_name
            self.index[name] = self
            self._info       = info
        def __repr__(self):
            return f"torch.{self.name}"
        def to_complex(self):
            return self.index[self.complex_name]
        def to_real(self):
            return self.index[self.real_name]

    class finfo:
        def __new__(cls, dtype_obj: NotTorch.dtype):
            info = dtype_obj._info
            if isinstance(info, _finfo):
                return info
            raise TypeError("torch.finfo() requires a floating point input type. Use torch.iinfo to handle 'torch.finfo'")

    class iinfo:
        def __new__(cls, dtype_obj: NotTorch.dtype):
            info = dtype_obj._info
            if isinstance(info, _iinfo):
                return info
            raise TypeError("torch.iinfo() requires an integer input type. Use torch.finfo to handle 'torch.iinfo'")

    class device:
        _index = {}
        def __init__(self, type):
            self._type = type
            self._index[type] = self
        def __repr__(self):
            return f"device(type={self._type!r})"
        def __str__(self):
            return self._type

    class layout:
        def __init__(self, name):
            self._name = name
        def __repr__(self):
            return f"torch.{self._name})"

    class memory_format:
        def __init__(self, name):
            self._name = name
        def __repr__(self):
            return f"torch.{self._name})"
        def __str__(self):
            return self._name

    class Size(tuple):
        def __repr__(self):
            size = ', '.join(map(str, self))
            return f"torch.Size([{size}])"
        def __setitem__(self):
            raise TypeError("'torch.Size' object does not support item assignment")
        def numel(self):
            mul = 1
            for i in self:
                mul *= i
            return mul

    float4_e2m1fn_x2 = dtype('float4_e2m1fn_x2', 0, 1, 1, 1, None, 'float4_e2m1fn_x2', _finfo._maker(bits=8, resolution=..., min=..., max=..., eps=..., smallest_normal=..., tiny=..., dtype=...))
    float8_e4m3fn    = dtype('float8_e4m3fn',    0, 1, 1, 1, None, 'float8_e4m3fn',    _finfo._maker(bits=8, resolution=1.0, min=-448.0, max=448.0, eps=0.125, smallest_normal=0.015625, tiny=0.015625, dtype='float8_e4m3fn'))
    float8_e4m3fnuz  = dtype('float8_e4m3fnuz',  0, 1, 1, 1, None, 'float8_e4m3fnuz',  _finfo._maker(bits=8, resolution=1.0, min=-240.0, max=240.0, eps=0.125, smallest_normal=0.0078125, tiny=0.0078125, dtype='float8_e4m3fnuz'))
    float8_e5m2      = dtype('float8_e5m2',      0, 1, 1, 1, None, 'float8_e5m2',      _finfo._maker(bits=8, resolution=1.0, min=-57344.0, max=57344.0, eps=0.25, smallest_normal=6.103515625e-05, tiny=6.103515625e-05, dtype='float8_e5m2'))
    float8_e5m2fnuz  = dtype('float8_e5m2fnuz',  0, 1, 1, 1, None, 'float8_e5m2fnuz',  _finfo._maker(bits=8, resolution=1.0, min=-57344.0, max=57344.0, eps=0.125, smallest_normal=3.0517578125e-05, tiny=3.0517578125e-05, dtype='float8_e5m2fnuz'))
    float8_e8m0fnu   = dtype('float8_e8m0fnu',   0, 1, 0, 1, None, 'float8_e8m0fnu',   _finfo._maker(bits=8, resolution=1.0, min=5.877471754111438e-39, max=1.7014118346046923e+38, eps=1.0, smallest_normal=5.877471754111438e-39, tiny=5.877471754111438e-39, dtype='float8_e8m0fnu'))
    bfloat16   = dtype('bfloat16',   0, 1, 1, 2, 'complex64', 'bfloat16',  _finfo._maker(bits=16, resolution=0.01, min=-3.3895313892515355e+38, max=3.3895313892515355e+38, eps=0.0078125, smallest_normal=1.1754943508222875e-38, tiny=1.1754943508222875e-38, dtype='bfloat16'))
    bit        = dtype('bit',        0, 0, 0, 1, None, 'uint1')
    bits16     = dtype('bits16',     0, 0, None, 2, None, 'bits16')
    bits1x8    = dtype('bits1x8',    0, 0, None, 1, None, 'bits1x8')
    bits2x4    = dtype('bits2x4',    0, 0, None, 1, None, 'bits2x4')
    bits4x2    = dtype('bits4x2',    0, 0, None, 1, None, 'bits4x2')
    bits8      = dtype('bits8',      0, 0, None, 1, None, 'bits8')
    _bool      = dtype('bool',       0, 0, 0,  1, None, 'bool')
    cdouble    = dtype('cdouble',    1, 0, 1, 16, 'complex128', 'float64', _finfo._maker(bits=64, resolution=1e-15, min=-1.7976931348623157e+308, max=1.7976931348623157e+308, eps=2.220446049250313e-16, smallest_normal=2.2250738585072014e-308, tiny=2.2250738585072014e-308, dtype='float64'))
    cfloat     = dtype('cfloat',     1, 0, 1,  8, 'complex64',  'float32', _finfo._maker(bits=32, resolution=1e-06, min=-3.4028234663852886e+38, max=3.4028234663852886e+38, eps=1.1920928955078125e-07, smallest_normal=1.1754943508222875e-38, tiny=1.1754943508222875e-38, dtype='float32'))
    chalf      = dtype('chalf',      1, 0, 1,  4, 'complex32',  'float16', _finfo._maker(bits=16, resolution=0.001, min=-65504.0, max=65504.0, eps=0.0009765625, smallest_normal=6.103515625e-05, tiny=6.103515625e-05, dtype='float16'))
    complex32  = dtype('complex32',  1, 0, 1,  4, 'complex32',  'float16', _finfo._maker(bits=16, resolution=0.001, min=-65504.0, max=65504.0, eps=0.0009765625, smallest_normal=6.103515625e-05, tiny=6.103515625e-05, dtype='float16'))
    complex64  = dtype('complex64',  1, 0, 1,  8, 'complex64',  'float32', _finfo._maker(bits=32, resolution=1e-06, min=-3.4028234663852886e+38, max=3.4028234663852886e+38, eps=1.1920928955078125e-07, smallest_normal=1.1754943508222875e-38, tiny=1.1754943508222875e-38, dtype='float32'))
    complex128 = dtype('complex128', 1, 0, 1, 16, 'complex128', 'float64', _finfo._maker(bits=64, resolution=1e-15, min=-1.7976931348623157e+308, max=1.7976931348623157e+308, eps=2.220446049250313e-16, smallest_normal=2.2250738585072014e-308, tiny=2.2250738585072014e-308, dtype='float64'))
    double     = dtype('double',     0, 1, 1,  8, 'complex128', 'float64', _finfo._maker(bits=64, resolution=1e-15, min=-1.7976931348623157e+308, max=1.7976931348623157e+308, eps=2.220446049250313e-16, smallest_normal=2.2250738585072014e-308, tiny=2.2250738585072014e-308, dtype='float64'))
    _float     = dtype('float',      0, 1, 1,  4, 'complex64', 'float32',  _finfo._maker(bits=32, resolution=1e-06, min=-3.4028234663852886e+38, max=3.4028234663852886e+38, eps=1.1920928955078125e-07, smallest_normal=1.1754943508222875e-38, tiny=1.1754943508222875e-38, dtype='float32'))
    float16    = dtype('float16',    0, 1, 1,  2, 'complex32', 'float16',  _finfo._maker(bits=16, resolution=0.001, min=-65504.0, max=65504.0, eps=0.0009765625, smallest_normal=6.103515625e-05, tiny=6.103515625e-05, dtype='float16'))
    float32    = dtype('float32',    0, 1, 1,  4, 'complex64', 'float32',  _finfo._maker(bits=32, resolution=1e-06, min=-3.4028234663852886e+38, max=3.4028234663852886e+38, eps=1.1920928955078125e-07, smallest_normal=1.1754943508222875e-38, tiny=1.1754943508222875e-38, dtype='float32'))
    float64    = dtype('float64',    0, 1, 1,  8, 'complex128', 'float64', _finfo._maker(bits=64, resolution=1e-15, min=-1.7976931348623157e+308, max=1.7976931348623157e+308, eps=2.220446049250313e-16, smallest_normal=2.2250738585072014e-308, tiny=2.2250738585072014e-308, dtype='float64'))
    half       = dtype('half',       0, 1, 1,  2, 'complex32', 'float16',  _finfo._maker(bits=16, resolution=0.001, min=-65504.0, max=65504.0, eps=0.0009765625, smallest_normal=6.103515625e-05, tiny=6.103515625e-05, dtype='float16'))
    _int       = dtype('int',        0, 0, 1, 4, None, 'int32', _iinfo._maker(bits=32, min=-2147483648, max=2147483647, dtype='int32'))
    int1       = dtype('int1',       0, 0, 1, 1, None, 'int1')
    int2       = dtype('int2',       0, 0, 1, 1, None, 'int2')
    int3       = dtype('int3',       0, 0, 1, 1, None, 'int3')
    int4       = dtype('int4',       0, 0, 1, 1, None, 'int4')
    int5       = dtype('int5',       0, 0, 1, 1, None, 'int5')
    int6       = dtype('int6',       0, 0, 1, 1, None, 'int6')
    int7       = dtype('int7',       0, 0, 1, 1, None, 'int7')
    int8       = dtype('int8',       0, 0, 1, 1, None, 'int8',  _iinfo._maker(bits=8, min=-128, max=127, dtype='int8'))
    int16      = dtype('int16',      0, 0, 1, 2, None, 'int16', _iinfo._maker(bits=16, min=-32768, max=32767, dtype='int16'))
    int32      = dtype('int32',      0, 0, 1, 4, None, 'int32', _iinfo._maker(bits=32, min=-2147483648, max=2147483647, dtype='int32'))
    int64      = dtype('int64',      0, 0, 1, 8, None, 'int64', _iinfo._maker(bits=64, min=-9223372036854775808, max=9223372036854775807, dtype='int64'))
    long       = dtype('long',       0, 0, 1, 8, None, 'int64', _iinfo._maker(bits=64, min=-9223372036854775808, max=9223372036854775807, dtype='int64'))
    qint8      = dtype('qint8',      0, 0, None, 1, None, 'qint8',    _iinfo._maker(bits=8, min=-128, max=127, dtype=...))
    qint32     = dtype('qint32',     0, 0, None, 4, None, 'qint32',   _iinfo._maker(bits=32, min=-2147483648, max=2147483647, dtype=...))
    quint2x4   = dtype('quint2x4',   0, 0, None, 1, None, 'quint2x4', _iinfo._maker(bits=8, min=0, max=255, dtype=...))
    quint4x2   = dtype('quint4x2',   0, 0, None, 1, None, 'quint4x2', _iinfo._maker(bits=8, min=0, max=255, dtype=...))
    quint8     = dtype('quint8',     0, 0, None, 1, None, 'quint8',   _iinfo._maker(bits=8, min=0, max=255, dtype=...))
    short      = dtype('short',      0, 0, 1, 2, None, 'int16', _iinfo._maker(bits=16, min=-32768, max=32767, dtype='int16'))
    uint1      = dtype('uint1',      0, 0, 0, 1, None, 'uint1')
    uint2      = dtype('uint2',      0, 0, 0, 1, None, 'uint2')
    uint3      = dtype('uint3',      0, 0, 0, 1, None, 'uint3')
    uint4      = dtype('uint4',      0, 0, 0, 1, None, 'uint4')
    uint5      = dtype('uint5',      0, 0, 0, 1, None, 'uint5')
    uint6      = dtype('uint6',      0, 0, 0, 1, None, 'uint6')
    uint7      = dtype('uint7',      0, 0, 0, 1, None, 'uint7')
    uint8      = dtype('uint8',      0, 0, 0, 1, None, 'uint8',  _iinfo._maker(bits=8, min=0, max=255, dtype='uint8'))
    uint16     = dtype('uint16',     0, 0, 0, 2, None, 'uint16', _iinfo._maker(bits=16, min=0, max=65535, dtype='uint16'))
    uint32     = dtype('uint32',     0, 0, 0, 4, None, 'uint32', _iinfo._maker(bits=32, min=0, max=4294967295, dtype='uint32'))
    uint64     = dtype('uint64',     0, 0, 0, 8, None, 'uint64', _iinfo._maker(bits=64, min=0, max=18446744073709551615, dtype='uint64'))

  # print(finfo(uint64))           # TypeError: torch.finfo() requires a floating point input type. Use torch.iinfo to handle 'torch.finfo'
  # print(finfo(float4_e2m1fn_x2)) # TypeError: bad argument type for built-in operation
  # print(finfo(float8_e4m3fn))    # finfo(bits=8, resolution=1.0, min=-448.0, max=448.0, eps=0.125, smallest_normal=0.015625, tiny=0.015625, dtype='float8_e4m3fn')

  # print(iinfo(float64)) # TypeError: torch.iinfo() requires an integer input type. Use torch.finfo to handle 'torch.iinfo'
  # print(iinfo(qint8))   # TypeError: bad argument type for built-in operation
  # print(iinfo(uint8))   # iinfo(bits=8, min=0, max=255, dtype='uint8')

    cpu = device("cpu")
    npu = device("npu")

    strided    = layout("strided")
    jagged     = layout("jagged")
    _mkldnn    = layout("_mkldnn")
    sparse_bsc = layout("sparse_bsc")
    sparse_bsr = layout("sparse_bsr")
    sparse_coo = layout("sparse_coo")
    sparse_csc = layout("sparse_csc")
    sparse_csr = layout("sparse_csr")

    channels_last     = memory_format('channels_last')
    channels_last_3d  = memory_format('channels_last_3d')
    contiguous_format = memory_format('contiguous_format')
    contiguous_format = memory_format('contiguous_format')
    preserve_format   = memory_format('preserve_format')

    # types from: /opt/python311/lib/python3.11/site-packages/torch/_C/_VariableFunctions.pyi

    ShapeType:        TypeAlias = Union[torch.Size, list[int], tuple[int, ...]]
    StrideType:       TypeAlias = Union[list[int], tuple[int, ...]]
    DimsType:         TypeAlias = Union[int, list[int], tuple[int, ...]]
    DimsSequenceType: TypeAlias = Union[list[int], tuple[int, ...]]
    # TODO: Type[torch.SymInt], Type[torch.SymFloat]
    NumberTypeType:   TypeAlias = Union[type[bool], type[int], type[float], type[complex]]
    # TODO: This needs a lot more type annotations
    # NumberType = Union[bool, int, float, complex, torch.SymInt, torch.SymFloat]
    NumberType:       TypeAlias = Union[bool, int, float, complex]
    RealNumberType:   TypeAlias = Union[bool, int, float]

    Number = (bool, int, float, complex, torch.SymInt, torch.SymFloat, torch.SymBool)
    # I don't call it Integral because numbers.Integral includes bool, but IntLike
    # does not
    Dim = int
    IntLike = (int, torch.SymInt)
    FloatLike = (float, torch.SymFloat)
    BoolLike = (bool, torch.SymBool)
    IntWithoutSymInt = int
    FloatWithoutSymFloat = float
    DeviceLikeType: TypeAlias = Union[str, torch.device, int]

    def tensor(
        data:          Any,
        dtype:         dtype | None = None,
        device:        DeviceLikeType | None = None,
        requires_grad: bool = False,
        pin_memory:    bool = False,
    ) -> Tensor:
        real = torch.tensor(data, dtype, device, requires_grad, pin_memory)
        return NotTorch.Tensor(real)

    class Tensor:
        _real_tensor_str = torch._tensor_str
        def __init__(self, real_tensor, real_device=None):
            assert str(real_tensor.device).startswith("cpu")
            self._tensor = real_tensor
            self.is_cpu = real_device is None or real_device.startswith("cpu")

        def __repr__(self):
            torch._tensor_str = self._real_tensor_str
            base = repr(self._tensor)
            del torch._tensor_str
            if self.is_cpu:
                return base
            assert base.endswith(")")
            return f"{base[:-1]}, device={self._device!r})"

    class DoubleTensor(Tensor):   ...
    class FloatTensor(Tensor):    ...
    class BFloat16Tensor(Tensor): ...
    class LongTensor(Tensor):     ...
    class IntTensor(Tensor):      ...
    class ShortTensor(Tensor):    ...
    class HalfTensor(Tensor):     ...
    class CharTensor(Tensor):     ...
    class ByteTensor(Tensor):     ...
    class BoolTensor(Tensor):     ...

    autograd     = torch.autograd
    return_types = torch.return_types
    library      = torch.library
    nn           = torch.nn



def wrap_torch_function(name, fn):
    real_Tensor = torch.Tensor
    def wrapper(*args, **kwargs):
        # 1. Распаковываем позиционные аргументы (NotTorch.Tensor → torch.Tensor)
        new_args = []
        for a in args:
            if isinstance(a, NotTorch.Tensor):
                new_args.append(a._tensor)
            else:
                new_args.append(a)

        # 2. Распаковываем именованные аргументы (Tensor / dtype / device)
        new_kwargs = {}
        real_device = None
        for k, v in kwargs.items():
            # 2.1 Tensor
            if isinstance(v, NotTorch.Tensor):
                new_kwargs[k] = v._tensor
            elif k == "device":
                if isinstance(v, NotTorch.device):
                    v = v.name
                assert isinstance(v, str)
                assert v.startswith("cpu") or v.startswith("npu")
                real_device = v
            else:
                new_kwargs[k] = v

        # 3. Вызываем оригинальную функцию torch.<name>
        out = fn(*new_args, **new_kwargs)

        # 4. Оборачиваем одиночный тензор
        if isinstance(out, real_Tensor):
            return NotTorch.Tensor(out, real_device)

        # 5. Оборачиваем namedtuple (torch.return_types.*)
        if isinstance(out, tuple) and hasattr(out, "_fields"):
            return type(out)(*(NotTorch.Tensor(x, x.device) if isinstance(x, torch.Tensor) else x
                               for x in out))

        # 6. Оборачиваем списки и кортежи тензоров
        if isinstance(out, (tuple, list)):
            return type(out)(
                NotTorch.Tensor(x, x.device) if isinstance(x, torch.Tensor) else x
                for x in out
            )

        # 7. Всё остальное возвращаем как есть
        return out

    wrapper.__name__ = name
    return wrapper

def generate_all_wrappers():
    for name in ("randn",): # torch.__dict__.items():
        fn = getattr(torch, name)
        if callable(fn) and not name.startswith("_"):
            setattr(NotTorch, name, wrap_torch_function(name, fn))

generate_all_wrappers()



def DispatchKey_extractor():
    # pprint(torch._C.DispatchKey.__members__) # TypeError: unhashable type: 'instancemethod'
    members = {name: int(member) for name, member in torch._C.DispatchKey.__members__.items()}
    members = dict(sorted(members.items(), key=lambda kv: kv[1]))
    print(pformat(members, sort_dicts=False))
    exit()
# DispatchKey_extractor()

class NotTorch_C:
    class DispatchKeySet:
        def __init__(self, key=None):
            self.key = key
        def __repr__(self):
            return f"DispatchKeySet({self.key!r})"

    members = {
        'Undefined': 0,
        'Dense': 1,
        'Quantized': 5,
        'Sparse': 8,
        'SparseCsr': 9,
        'NestedTensor': 10,
        'BackendSelect': 11,
        'Python': 12,
        'FuncTorchDynamicLayerBackMode': 14,
        'Functionalize': 15,
        'Conjugate': 17,
        'Negative': 18,
        'ZeroTensor': 19,
        'ADInplaceOrView': 20,
        'AutogradOther': 21,
        'AutogradFunctionality': 22,
        'AutogradNestedTensor': 23,
        'AutocastCPU': 25,
        'AutocastXPU': 28,
        'AutocastIPU': 29,
        'AutocastHPU': 30,
        'AutocastMPS': 32,
        'AutocastCUDA': 33,
        'AutocastPrivateUse1': 34,
        'FuncTorchBatched': 35,
        'FuncTorchVmapMode': 37,
        'FuncTorchGradWrapper': 40,
        'PythonTLSSnapshot': 42,
        'FuncTorchDynamicLayerFrontMode': 43,
        'PreDispatch': 46,
        'PythonDispatcher': 47,
        'StartOfDenseBackends': 49,
        'CPU': 50,
        'CUDA': 51,
        'HIP': 52,
        'XLA': 53,
        'MPS': 54,
        'IPU': 55,
        'XPU': 56,
        'HPU': 57,
        'VE': 58,
        'Lazy': 59,
        'MTIA': 60,
        'MAIA': 61,
        'PrivateUse1': 62,
        'PrivateUse2': 63,
        'PrivateUse3': 64,
        'Meta': 65,
        'EndOfDenseBackends': 65,
        'StartOfQuantizedBackends': 66,
        'QuantizedCPU': 67,
        'QuantizedCUDA': 68,
        'QuantizedHIP': 69,
        'QuantizedXLA': 70,
        'QuantizedMPS': 71,
        'QuantizedIPU': 72,
        'QuantizedXPU': 73,
        'QuantizedHPU': 74,
        'QuantizedVE': 75,
        'QuantizedLazy': 76,
        'QuantizedMTIA': 77,
        'QuantizedMAIA': 78,
        'QuantizedPrivateUse1': 79,
        'QuantizedPrivateUse2': 80,
        'QuantizedPrivateUse3': 81,
        'QuantizedMeta': 82,
        'EndOfQuantizedBackends': 82,
        'StartOfSparseBackends': 83,
        'SparseCPU': 84,
        'SparseCUDA': 85,
        'SparseHIP': 86,
        'SparseXLA': 87,
        'SparseMPS': 88,
        'SparseIPU': 89,
        'SparseXPU': 90,
        'SparseHPU': 91,
        'SparseVE': 92,
        'SparseLazy': 93,
        'SparseMTIA': 94,
        'SparseMAIA': 95,
        'SparsePrivateUse1': 96,
        'SparsePrivateUse2': 97,
        'SparsePrivateUse3': 98,
        'SparseMeta': 99,
        'EndOfSparseBackends': 99,
        'StartOfSparseCsrBackends': 100,
        'SparseCsrCPU': 101,
        'SparseCsrCUDA': 102,
        'SparseCsrHIP': 103,
        'SparseCsrXLA': 104,
        'SparseCsrMPS': 105,
        'SparseCsrIPU': 106,
        'SparseCsrXPU': 107,
        'SparseCsrHPU': 108,
        'SparseCsrVE': 109,
        'SparseCsrLazy': 110,
        'SparseCsrMTIA': 111,
        'SparseCsrMAIA': 112,
        'SparseCsrPrivateUse1': 113,
        'SparseCsrPrivateUse2': 114,
        'SparseCsrPrivateUse3': 115,
        'SparseCsrMeta': 116,
        'EndOfSparseCsrBackends': 116,
        'StartOfNestedTensorBackends': 117,
        'NestedTensorCPU': 118,
        'NestedTensorCUDA': 119,
        'NestedTensorHIP': 120,
        'NestedTensorXLA': 121,
        'NestedTensorMPS': 122,
        'NestedTensorIPU': 123,
        'NestedTensorXPU': 124,
        'NestedTensorHPU': 125,
        'NestedTensorVE': 126,
        'NestedTensorLazy': 127,
        'NestedTensorMTIA': 128,
        'NestedTensorMAIA': 129,
        'NestedTensorPrivateUse1': 130,
        'NestedTensorPrivateUse2': 131,
        'NestedTensorPrivateUse3': 132,
        'NestedTensorMeta': 133,
        'EndOfNestedTensorBackends': 133,
        'StartOfAutogradFunctionalityBackends': 134,
        'AutogradCPU': 135,
        'AutogradCUDA': 136,
        'AutogradHIP': 137,
        'AutogradXLA': 138,
        'AutogradMPS': 139,
        'AutogradIPU': 140,
        'AutogradXPU': 141,
        'AutogradHPU': 142,
        'AutogradVE': 143,
        'AutogradLazy': 144,
        'AutogradMTIA': 145,
        'AutogradMAIA': 146,
        'AutogradPrivateUse1': 147,
        'AutogradPrivateUse2': 148,
        'AutogradPrivateUse3': 149,
        'AutogradMeta': 150,
        'EndOfAutogradFunctionalityBackends': 150,
        'Autograd': 151,
        'CompositeImplicitAutograd': 152,
        'FuncTorchBatchedDecomposition': 153,
        'CompositeImplicitAutogradNestedTensor': 154,
        'CompositeExplicitAutograd': 155,
        'CompositeExplicitAutogradNonFunctional': 156
    }
    DispatchKey = IntEnum('DispatchKey', members)

    _dispatch_library = torch._C._dispatch_library



class NotTorchSwitcher:
    def __init__(self):
        not_module = types.ModuleType("torch")
        for attr, value in NotTorch.__dict__.items():
            if not attr.startswith("__"):
                setattr(not_module, attr, value)

        not_module.bool  = NotTorch._bool
        not_module.int   = NotTorch._int
        not_module.float = NotTorch._float
        not_module.npu   = torch.npu
        not_module.__version__ = '2.11.0+cpu'

        self.not_module  = not_module.__dict__
        self.real_module = dict(torch.__dict__)

        not_C_module = types.ModuleType("torch._C")
        for attr, value in NotTorch_C.__dict__.items():
            if not attr.startswith("__"):
                setattr(not_C_module, attr, value)

        self.not_C_module  = not_C_module
        self.real_C_module = torch._C

    def on(self):
        torch.__dict__.clear()
        torch.__dict__.update(self.not_module)
        torch._C = self.not_C_module

    def off(self):
        torch.__dict__.clear()
        torch.__dict__.update(self.real_module)
        torch._C = self.real_C_module

not_torch_switcher = NotTorchSwitcher()
not_torch_switcher.on()

# print(torch.randn(2, 3))
print(torch.randn(2, 3, device="npu"))
exit()
