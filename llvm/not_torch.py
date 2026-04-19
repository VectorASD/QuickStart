from __future__ import annotations

import torch
from triton.runtime.cache import FileCacheManager

import os
import inspect
import types, sys
from pprint import pformat
import warnings

from typing import TYPE_CHECKING



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
    def get_device_properties(device_id):
        # TODO: это лишь угаданные значения
        return {
            "max_shared_memory": 0,
            "max_shared_memory_per_multiprocessor": 0,
        }

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

def DispatchKey_extractor():
    # pprint(torch._C.DispatchKey.__members__) # TypeError: unhashable type: 'instancemethod'
    members = {name: int(member) for name, member in torch._C.DispatchKey.__members__.items()}
    members = dict(sorted(members.items(), key=lambda kv: kv[1]))
    print(pformat(members, sort_dicts=False))
    exit()
# DispatchKey_extractor()



#    Исправляет подсветку
# Для анализатора типов: TYPE_CHECKING == True
# Для runtime:           TYPE_CHECKING == False
if TYPE_CHECKING:
    import torch as _torch
else:
    _torch = types.ModuleType("torch")
    _torch.__dict__.update(torch.__dict__)
    _torch._C = types.ModuleType("torch._C")
    _torch._C.__dict__.update(torch._C.__dict__)



class device:
    type: str
    index: int | None

    def __new__(cls, type: str | device, index: int = None):
        if isinstance(type, device):
            assert index is None
            return type

        if isinstance(type, _torch.device):
            assert index is None
            type, index = type.type, type.index

        if index is None:
            assert isinstance(type, str)
            if ":" in type:
                type, index = type.split(":", 1)
                index = int(index)
        else:
            assert isinstance(index, int)
        obj = super().__new__(cls)
        obj.type = type
        obj.index = index
        return obj

    def __repr__(self):
        if self.index is not None:
            return f"device(type={self.type!r}, index={self.index})"
        return f"device(type={self.type!r})"

    def __str__(self):
        if self.index is not None:
            return f"{self.type}:{self.index}"
        return self.type

    @property
    def _index(self):
        if self.index is None:
            return 0
        return self.index

    def __eq__(self, other):
        if not isinstance(other, device):
            return NotImplemented
        return self.type == other.type and self._index == other._index

    def __ne__(self, other):
        if not isinstance(other, device):
            return NotImplemented
        return self.type != other.type or self._index != other._index

    def __hash__(self):
        return hash((self.type, self.index))



class Tensor:
    def __init__(self, real_tensor, real_device=None):
        assert real_tensor.device.type == "cpu"
        self._tensor = real_tensor
        if real_device is None:
            real_device = device("cpu")
        self._device = real_device

    def __repr__(self, *, tensor_contents=None):
        if _torch.overrides.has_torch_function_unary(self):
            return _torch.overrides.handle_torch_function(
                _torch.Tensor.__repr__, (self,), self, tensor_contents=tensor_contents
            )
        # All strings are unicode in Python 3.
        return _torch._tensor_str._str(self, tensor_contents=tensor_contents)

    @property
    def device(self):
        return self._device

    def to(self, *args, **kwargs) -> Tensor:
        """
        Поддерживает три формы вызова:
            (                        dtype: _dtype = None, non_blocking: _bool = False, copy: _bool = False, *, memory_format: torch.memory_format | None = None)
            (device: DeviceLikeType, dtype: _dtype = None, non_blocking: _bool = False, copy: _bool = False, *, memory_format: torch.memory_format | None = None)
            (other: Tensor,                                non_blocking: _bool = False, copy: _bool = False, *, memory_format: torch.memory_format | None = None)
        """
        device_arg    = kwargs.get("device", None)
        dtype_arg     = kwargs.get("dtype", None)
        non_blocking  = kwargs.get("non_blocking", False)
        copy          = kwargs.get("copy", False)
        memory_format = kwargs.get("memory_format", None)

        if args and isinstance(args[0], Tensor):
            other = args[0]
            device_arg = other.device
            dtype_arg  = other._tensor.dtype
            if len(args) > 1: non_blocking = args[1]
            if len(args) > 2: copy         = args[2]
        elif args and isinstance(args[0], (device, str)):
            device_arg = args[0]
            if len(args) > 1: dtype_arg    = args[1]
            if len(args) > 2: non_blocking = args[2]
            if len(args) > 3: copy         = args[3]
        else:
            if len(args) > 0: dtype_arg    = args[0]
            if len(args) > 1: non_blocking = args[1]
            if len(args) > 2: copy         = args[2]

        real_tensor = self._tensor
        new_device  = self._device if device_arg is None else device(device_arg)

        if dtype_arg is not None:
            real_tensor = real_tensor.to(dtype=dtype_arg)
        if copy:
            real_tensor = real_tensor.clone()
        if memory_format is not None:
            real_tensor = real_tensor.contiguous(memory_format=memory_format)
        return Tensor(real_tensor, real_device=new_device)

    def __getattr__(self, name):
        return getattr(self._tensor, name)

    def _binary(self, other, op):
        if isinstance(other, Tensor):
            if self._device != other._device:
                raise RuntimeError(f"Expected both tensors to be on the same device, got {self._device} and {other._device}")
            other = other._tensor
        return Tensor(op(self._tensor, other), self._device)

    def __and__(self, other):
        return self._binary(other, lambda a, b: a & b)

    def __or__(self, other):
        return self._binary(other, lambda a, b: a | b)

    def __xor__(self, other):
        return self._binary(other, lambda a, b: a ^ b)

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    def __mul__(self, other):
        return self._binary(other, lambda a, b: a * b)

    def __iter__(self):
        if self._tensor.dim() == 0:
            raise TypeError("iteration over a 0-d tensor")
        self._iter_index = 0
        return self

    def __next__(self):
        if self._iter_index >= self._tensor.size(0):
            raise StopIteration
        item = self._tensor[self._iter_index]
        self._iter_index += 1
        return Tensor(item, self._device)

    def __format__(self, format_spec):
        if self._tensor.numel() == 1:
            return format(self._tensor.item(), format_spec)

        raise TypeError(
            "format() is only supported for scalar tensors; "
           f"got tensor with shape {tuple(self._tensor.shape)}"
        )

    def __getitem__(self, idx):
        return Tensor(self._tensor[idx], self._device)

"""
from typing import Any, TypeAlias, Union

DeviceLikeType: TypeAlias = Union[str, torch.device, int]

def tensor(
    data:          Any,
    dtype:         torch.dtype | None = None,
    device:        DeviceLikeType | None = None,
    requires_grad: bool = False,
    pin_memory:    bool = False,
) -> Tensor:
    real = torch.tensor(data, dtype, device, requires_grad, pin_memory)
    return Tensor(real)

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
"""



import functools
import contextlib

SKIPED_TYPES = {
    bool,
    int,
    str,
    torch.dtype,
    contextlib._GeneratorContextManager,
    torch._ops._ModeStackStateForPreDispatch,
    torch._vendor.packaging._structures.InfinityType, # здесь всего 2 типа, без родительского общего класса
    torch._vendor.packaging._structures.NegativeInfinityType,
    torch.testing._comparison.TensorLikePair,
}
UnpackedDualTensor = torch.autograd.forward_ad.UnpackedDualTensor

def type_wrapper(obj, real_device):
    if obj is None or type(obj) in SKIPED_TYPES:
        return obj

    if type(obj) is _torch.Tensor:
        return Tensor(obj, real_device)

    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return wrap_torch_function(obj)

    if isinstance(obj, UnpackedDualTensor):
        return type(obj)(*(type_wrapper(v, real_device) for v in obj))
    if isinstance(obj, tuple):
        return tuple(type_wrapper(v, real_device) for v in obj)
    if isinstance(obj, list):
        return [type_wrapper(v, real_device) for v in obj]

    print("[my type_wrapper] unsupported type:", type(obj))
    exit()

def wrap_torch_function(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        fn_name = fn.__name__ # отдельная переменная, т.к. отладчик не видит closure
        print("[RUN]", fn_name)
        # 1. Распаковываем позиционные аргументы (NotTorch.Tensor → torch.Tensor)
        new_args = []
        founded_device = None
        for a in args:
            if isinstance(a, Tensor):
                new_args.append(a._tensor)
                if founded_device is None:
                    founded_device = a._device
                elif founded_device != a._device:
                    raise RuntimeError(f"Device mismatch: {founded_device} vs {a._device}")
            else:
                new_args.append(a)

        # 2. Распаковываем именованные аргументы (Tensor / dtype / device)
        new_kwargs = {}
        real_device = None
        for k, v in kwargs.items():
            # 2.1 Tensor
            if isinstance(v, Tensor):
                new_kwargs[k] = v._tensor
                if founded_device is None:
                    founded_device = v._device
                elif founded_device != v._device:
                    raise RuntimeError(f"Device mismatch: {founded_device} vs {v._device}")
            elif k == "device":
                if not isinstance(v, device):
                    v = device(v)
                assert v.type in ("cpu", "npu")
                real_device = v
            else:
                new_kwargs[k] = v

        if real_device is None:
            real_device = founded_device

        # 3. Вызываем оригинальную функцию torch.<name>
        out = fn(*new_args, **new_kwargs)

        # 4. Оборачиваем результат
        return type_wrapper(out, real_device)
    return wrapper



def wrapper_bot(module, visited):
    # print(module.__name__)
    visited[module] = wrapper = types.ModuleType(module.__name__)

    for attr_name, value in module.__dict__.items():
        if inspect.ismodule(value):
            if value.__name__.startswith("torch") and value.__name__ != "torch._tensor_str":
                try:
                    wrapped = visited[value]
                except KeyError:
                    wrapped = wrapper_bot(value, visited)
            else:
                wrapped = value
        elif attr_name.startswith("__"):
            wrapped = value
        elif inspect.isfunction(value) or inspect.isbuiltin(value):
            # print(" ", type(value), value.__module__, attr_name, value)
            wrapped = wrap_torch_function(value)
        else:
            wrapped = value

        setattr(wrapper, attr_name, wrapped)

    return wrapper

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    visited = {}
    wrapper_bot(torch, visited)

visited[torch].Tensor = Tensor
visited[torch].device = device



import not_aten

# def wrap_torch_function(fn):
#    @functools.wraps(fn)
#    def wrapper(*args, **kwargs):
#        out = nottorch_dispatch(fn, args, kwargs)
#        return type_wrapper(out)
#
#    return wrapper

_reserved_namespaces = ["prim"]
_impls: set[str] = set()

class Library:
    def __init__(self, ns, kind, dispatch_key=""):
        if kind not in ("IMPL", "DEF", "FRAGMENT"):
            raise ValueError("Unsupported kind: ", kind)
        if ns in _reserved_namespaces and (kind == "DEF" or kind == "FRAGMENT"):
            raise ValueError(ns, " is a reserved namespace. Please try creating a library with another name.")
        self.ns           = ns
        self.kind         = kind
        self.dispatch_key = dispatch_key
        print("[LIB INIT]", self)
        self._op_impls: set[str] = set()

    def __repr__(self):
        return f"Library(kind={self.kind}, ns={self.ns}, dispatch_key={self.dispatch_key})>"
        # разрабы потеряли '<' в начале... емаё!!!)))

    def impl(self, op_name, fn, dispatch_key="", *, with_keyset=False, allow_override=False):
        if not callable(fn):
            raise TypeError(f"Input function is required to be a callable but found type {type(fn)}")
        if dispatch_key == "":
            dispatch_key = self.dispatch_key

        if isinstance(op_name, str):
            name = op_name
        elif isinstance(op_name, torch._ops.OpOverload):
            name = op_name._schema.name
            overload_name = op_name._schema.overload_name
            if overload_name != "":
                name = f"{name}.{overload_name}"
        else:
            raise RuntimeError("impl should be passed either a name or an OpOverload object as the first argument")

        key = f"{self.ns}/{name.split('::')[-1]}/{dispatch_key}"
        if (not allow_override) and key in _impls:
            # TODO: in future, add more info about where the existing function is registered (this info is
            # today already returned by the C++ warning when impl is called but we error out before that)
            raise RuntimeError(
                "This is not allowed since there's already a kernel registered from python overriding {}"
                "'s behavior for {} dispatch key and {} namespace.".format(
                    name.split("::")[-1], dispatch_key, self.ns
                )
            )

        if dispatch_key == "": dispatch_key = "CompositeImplicitAutograd"

        assert dispatch_key == "PrivateUse1"
        assert with_keyset == False
        assert allow_override == False
        print("[LIB IMPL]", key, fn)

        _impls.add(key)
        self._op_impls.add(key)

    def _destroy(self):
        global _impls
        _impls -= self._op_impls
        print("[LIB DESTROY]")
        """   имеет смысла, если бы нужен был 'define':
        for name in self._op_defs:
            # Delete the cached torch.ops.ns.foo if it was registered.
            # Otherwise, accessing it leads to a segfault.
            # It's possible that we only registered an overload in this Library
            # and another library owns an alive overload.
            # That's OK - the next time torch.ops.ns.foo gets called, it'll be
            # recomputed to point at the right collection of overloads.
            ns, name_with_overload = name.split("::")
            name = name_with_overload.split(".")[0]
            if not hasattr(torch.ops, ns):
                continue
            namespace = getattr(torch.ops, ns)
            if not hasattr(namespace, name):
                continue
            delattr(namespace, name)
            namespace._dir.remove(name)
        """

not_library = types.ModuleType("torch.library")
not_library.Library = Library
visited[torch].library = not_library



class NotTorchSwitcher:
    def __init__(self, visited):
        self.not_dicts  = {real_mod: not_mod.__dict__        for real_mod, not_mod in visited.items()}
        self.real_dicts = {real_mod: dict(real_mod.__dict__) for real_mod in visited}

    def on(self):
        for real_mod, not_dict in self.not_dicts.items():
            real_mod.__dict__.clear()
            real_mod.__dict__.update(not_dict)

    def off(self):
        for real_mod, real_dict in self.real_dicts.items():
            real_mod.__dict__.clear()
            real_mod.__dict__.update(real_dict)

not_torch_switcher = NotTorchSwitcher(visited)
not_torch_switcher.on()



# print(torch.randn(2, 3))
# print(torch.randn(2, 3, device="npu"))
# print(torch.randn(2, 3).to("npu"))
# print(torch.library.Library, Library)
# exit()
