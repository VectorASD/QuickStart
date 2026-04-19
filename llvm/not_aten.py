import torch
from torch.overrides import has_torch_function, handle_torch_function, _get_overloaded_args, has_torch_function_unary

from enum import IntEnum

class ParameterType(IntEnum):
    TENSOR           =  0
    SCALAR           =  1
    INT64            =  2
    SYM_INT          =  3
    DOUBLE           =  4
    COMPLEX          =  5
    TENSOR_LIST      =  6
    INT_LIST         =  7
    SYM_INT_LIST     =  8
    FLOAT_LIST       =  9
    GENERATOR        = 10
    BOOL             = 11
    STORAGE          = 12
    PYOBJECT         = 13
    SCALARTYPE       = 14
    LAYOUT           = 15
    MEMORY_FORMAT    = 16
    QSCHEME          = 17
    DEVICE           = 18
    STREAM           = 19
    STRING           = 20
    DIMNAME          = 21
    DIMNAME_LIST     = 22
    SCALAR_LIST      = 23
    DISPATCH_KEY_SET = 24

TYPE_MAP = {
    "Tensor":  ParameterType.TENSOR,
    "Scalar":  ParameterType.SCALAR,
    "int64_t": ParameterType.INT64,
    "SymInt":  ParameterType.SYM_INT,
    "double":  ParameterType.DOUBLE,
    "complex": ParameterType.COMPLEX,

    "TensorList":                         ParameterType.TENSOR_LIST,
    "c10::List<::std::optional<Tensor>>": ParameterType.TENSOR_LIST,

    "IntArrayRef":      ParameterType.INT_LIST,
    "SymIntArrayRef":   ParameterType.SYM_INT_LIST,
    "ArrayRef<double>": ParameterType.FLOAT_LIST,

    "Generator": ParameterType.GENERATOR,
    "bool":      ParameterType.BOOL,
    "Storage":   ParameterType.STORAGE,
    "PyObject*": ParameterType.PYOBJECT,

    "ScalarType":   ParameterType.SCALARTYPE,
    "Layout":       ParameterType.LAYOUT,
    "MemoryFormat": ParameterType.MEMORY_FORMAT,
    "QScheme":      ParameterType.QSCHEME,

    "Device":      ParameterType.DEVICE,
    "DeviceIndex": ParameterType.INT64,

    "Stream": ParameterType.STREAM,

    "std::string":        ParameterType.STRING,
    "c10::string_view":   ParameterType.STRING,
    "std::string_view":   ParameterType.STRING,
    "::std::string_view": ParameterType.STRING,

    "Dimname":     ParameterType.DIMNAME,
    "DimnameList": ParameterType.DIMNAME_LIST,

    "ScalarList":     ParameterType.SCALAR_LIST,
    "DispatchKeySet": ParameterType.DISPATCH_KEY_SET,
}



def _append_overloaded(overloaded_args, obj):
    obj_type = type(obj)

    # 1. Уже есть такой тип?
    for existing in overloaded_args:
        if type(existing) is obj_type:
            return

    # 2. Найти позицию для вставки (subclass → раньше superclass)
    insert_pos = len(overloaded_args)
    for i, existing in enumerate(overloaded_args):
        if issubclass(obj_type, type(existing)):
            insert_pos = i
            break

    overloaded_args.insert(insert_pos, obj)

def is_tensor_and_append_overloaded(obj, overloaded_args):
    # 1. Точный Tensor (как THPVariable_CheckExact)
    if type(obj) is torch.Tensor:
        return True

    # 2. Объект с __torch_function__ → добавить в overloaded_args
    tf = getattr(type(obj), "__torch_function__", None)
    if tf is not None and tf is not torch._C._disabled_torch_function_impl:
        _append_overloaded(overloaded_args, obj)
        return True

    # 3. Подкласс Tensor без __torch_function__
    if isinstance(obj, torch.Tensor):
        return True

    return False



def dtype_extractor():
    complex_arr = []
    floating_arr = []
    # signed_arr = []
    integer_arr = []
    for value in torch.__dict__.values():
        if isinstance(value, torch.dtype):
            if value.is_complex:        complex_arr.append(value)
            if value.is_floating_point: floating_arr.append(value)
            # try:
            #     if value.is_signed: signed_arr.append(value)
            # except RuntimeError: pass
            if not value.is_complex and not value.is_floating_point:
                integer_arr.append(value)
    print(complex_arr)
    print(floating_arr)
    print(integer_arr)
    exit()
# dtype_extractor()

COMPLEX_TYPES = {torch.complex32, torch.complex64, torch.complex128}
FLOAT_TYPES   = {torch.float16, torch.float32, torch.float64, torch.bfloat16, torch.float8_e5m2, torch.float8_e4m3fn, torch.float8_e5m2fnuz, torch.float8_e4m3fnuz, torch.float8_e8m0fnu, torch.float4_e2m1fn_x2}
INTEGER_TYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool, torch.qint8, torch.quint8, torch.qint32, torch.quint4x2, torch.quint2x4, torch.bits1x8, torch.bits2x4, torch.bits4x2, torch.bits8, torch.bits16, torch.uint16, torch.uint32, torch.uint64, torch.uint1, torch.uint2, torch.uint3, torch.uint4, torch.uint5, torch.uint6, torch.uint7, torch.int1, torch.int2, torch.int3, torch.int4, torch.int5, torch.int6, torch.int7}



def check_scalar(obj):
    return isinstance(obj, (int, float, complex, bool)) or (
        isinstance(obj, torch.Tensor) and
        obj.dim() == 0                and
        not obj.requires_grad
    )

def check_complex(obj):
    return isinstance(obj, complex) or (
        isinstance(obj, torch.Tensor) and
        obj.dim() == 0                and
        not obj.requires_grad         and
        obj.dtype in COMPLEX_TYPES
    )

def check_double(obj):
    return isinstance(obj, float) or (
        isinstance(obj, torch.Tensor) and
        obj.dim() == 0                and
        not obj.requires_grad         and
        obj.dtype in FLOAT_TYPES
    )

def check_integer(obj):
    return isinstance(obj, int) or (
        isinstance(obj, torch.Tensor) and
        obj.dim() == 0                and
        not obj.requires_grad         and
        obj.dtype in INTEGER_TYPES
    )



def is_tracing_enabled():
    return False

def is_int_or_symint_list(param, obj, overloaded_args, argnum, failed_idx):
    """
    Полный Python-порт C++ функции is_int_or_symint_list.
    """
    # --- 1. obj — tuple или list ---
    is_tuple = isinstance(obj, tuple)
    if is_tuple or isinstance(obj, list):
        size = len(obj)
        if size == 0:
            return True

        has_torch_func = False

        # --- 2. Проверяем каждый элемент ---
        for idx, item in enumerate(obj):

            # 2.1. torch_function?
            if overloaded_args is not None and has_torch_function_unary(item):
                overloaded_args.append(item)
                has_torch_func = True

            # 2.2. Первый элемент — строгая проверка типа
            if idx == 0:
                if isinstance(item, int):
                    continue

                # JIT tracer допускает scalar tensor как int
                r = (
                    is_tracing_enabled() and
                    isinstance(item, torch.Tensor) and
                    item.dim() == 0
                )

                if not r:
                    if failed_idx is not None:
                        failed_idx[0] = 0
                    if not has_torch_func:
                        return False

        return True

    # --- 3. Если obj — одиночный int, и broadcast_size > 0 ---
    broadcast_size = param.size
    if broadcast_size > 0 and isinstance(obj, int):
        return True

    return False

def is_tensor_list_and_append_overloaded(param, obj, overloaded_args, argnum, failed_idx):
    throw_error = True
    # must be list or tuple
    if not isinstance(obj, (list, tuple)):
        return False

    for idx, iobj in enumerate(obj):
        if not is_tensor_and_append_overloaded(iobj, overloaded_args):
            if throw_error:
                raise TypeError(
                    f"expected Tensor as element {idx} in argument {argnum}, "
                    f"but got {type(iobj).__name__}"
                )
            return False

    return True

def is_float_or_complex_list(param, obj, overloaded_args, argnum, failed_idx):
    # must be list or tuple
    if not isinstance(obj, (list, tuple)):
        return False

    has_torch_func = False

    for idx, iobj in enumerate(obj):
        # --- torch function override detection ---
        if overloaded_args is not None and has_torch_function_unary(iobj, ignore_mode=True):
            overloaded_args.append(iobj)
            has_torch_func = True

        # --- type check only for the first element ---
        if idx == 0 and not (check_complex(iobj) or check_double(iobj) or has_torch_func):
            return False

    return True



CHECK_TABLE = {
    ParameterType.TENSOR:           lambda param, obj, overloaded_args, argnum, failed_idx: is_tensor_and_append_overloaded(obj, overloaded_args) or (param.allow_numbers_as_tensors and check_scalar(obj)),
    ParameterType.SCALAR:           lambda param, obj, overloaded_args, argnum, failed_idx: check_scalar(obj),
    ParameterType.COMPLEX:          lambda param, obj, overloaded_args, argnum, failed_idx: check_complex(obj),
    ParameterType.DOUBLE:           lambda param, obj, overloaded_args, argnum, failed_idx: check_double(obj),
    ParameterType.INT64:            lambda param, obj, overloaded_args, argnum, failed_idx: check_integer(obj),
    ParameterType.SYM_INT:          lambda param, obj, overloaded_args, argnum, failed_idx: isinstance(obj, int),

    ParameterType.TENSOR_LIST:      is_tensor_list_and_append_overloaded,
    ParameterType.INT_LIST:         is_int_or_symint_list,
    ParameterType.SYM_INT_LIST:     is_int_or_symint_list,
    ParameterType.FLOAT_LIST:       is_float_or_complex_list,

    ParameterType.GENERATOR:        lambda param, obj, overloaded_args, argnum, failed_idx: isinstance(obj, torch.Generator),
    ParameterType.BOOL:             lambda param, obj, overloaded_args, argnum, failed_idx: isinstance(obj, bool),
    ParameterType.STORAGE:          lambda param, obj, overloaded_args, argnum, failed_idx: isinstance(obj, torch.storage._StorageBase),
    ParameterType.PYOBJECT:         lambda param, obj, overloaded_args, argnum, failed_idx: True,

    ParameterType.SCALARTYPE:       lambda param, obj, overloaded_args, argnum, failed_idx: isinstance(obj, torch.dtype),
    ParameterType.LAYOUT:           lambda param, obj, overloaded_args, argnum, failed_idx: isinstance(obj, torch.layout),
    ParameterType.MEMORY_FORMAT:    lambda param, obj, overloaded_args, argnum, failed_idx: isinstance(obj, torch.memory_format),
    ParameterType.QSCHEME:          lambda param, obj, overloaded_args, argnum, failed_idx: isinstance(obj, torch.qscheme),

    ParameterType.DEVICE:           lambda param, obj, overloaded_args, argnum, failed_idx: isinstance(obj, (torch.device, str, int)),
    ParameterType.STREAM:           lambda param, obj, overloaded_args, argnum, failed_idx: hasattr(obj, "cuda_stream") or obj.__class__.__name__ == "Stream",
    ParameterType.STRING:           lambda param, obj, overloaded_args, argnum, failed_idx: isinstance(obj, str),

    ParameterType.DIMNAME:          lambda param, obj, overloaded_args, argnum, failed_idx: obj is None or isinstance(obj, str),
    ParameterType.DIMNAME_LIST:     lambda param, obj, overloaded_args, argnum, failed_idx:
        isinstance(obj, (list, tuple)) and (
            not obj        or
            obj[0] is None or
            isinstance(obj[0], str)
        ) if param.size != 1 else (
            obj is None or
            isinstance(obj, str)
        ),

    ParameterType.SCALAR_LIST:      lambda param, obj, overloaded_args, argnum, failed_idx: isinstance(obj, (list, tuple)) and all(check_scalar(x) for x in obj),
    ParameterType.DISPATCH_KEY_SET: lambda param, obj, overloaded_args, argnum, failed_idx: isinstance(obj, torch._C.DispatchKeySet),
}
CHECK_TABLE = tuple(CHECK_TABLE[i] for i in range(max(ParameterType) + 1))

TYPE_NAME_MAP = {
    ParameterType.TENSOR:           "Tensor",
    ParameterType.SCALAR:           "Number",
    ParameterType.INT64:            "int",
    ParameterType.SYM_INT:          "int",
    ParameterType.DOUBLE:           "float",
    ParameterType.COMPLEX:          "complex",

    ParameterType.TENSOR_LIST:      "tuple of Tensors",
    ParameterType.INT_LIST:         "tuple of ints",
    ParameterType.SYM_INT_LIST:     "tuple of ints",
    ParameterType.FLOAT_LIST:       "tuple of floats", # на самом деле правильно: "tuple of floats or complex numbers", это уже баг оригинального aten

    ParameterType.GENERATOR:        "torch.Generator",
    ParameterType.BOOL:             "bool",
    ParameterType.STORAGE:          "torch.Storage",
    ParameterType.PYOBJECT:         "object",

    ParameterType.SCALARTYPE:       "torch.dtype",
    ParameterType.LAYOUT:           "torch.layout",
    ParameterType.MEMORY_FORMAT:    "torch.memory_format",
    ParameterType.QSCHEME:          "torch.qscheme",

    ParameterType.DEVICE:           "torch.device",
    ParameterType.STREAM:           "torch.cuda.Stream",
    ParameterType.STRING:           "str",

    ParameterType.DIMNAME:          "name",
    ParameterType.DIMNAME_LIST:     "tuple of names",

    ParameterType.SCALAR_LIST:      "tuple of Scalars",
    ParameterType.DISPATCH_KEY_SET: "DispatchKeySet",
}
TYPE_NAME_MAP = tuple(TYPE_NAME_MAP[i] for i in range(max(ParameterType) + 1))



# source: https://github.com/pytorch/pytorch/blob/50eea6cd/torch/csrc/utils/python_arg_parser.cpp#L1305-L1336
def parse_intlist_args(s: str, size: int):
    # case 0: empty string → empty list
    if not s:
        return []

    # case 1: scalar (e.g. "2")
    if s[0] != "{":
        if size <= 0:
            raise RuntimeError(f"Incorrect size of IntArrayRef: {size}")
        # repeat scalar 'size' times
        return [int(s)] * size

    # case 2: list (e.g. "{1,2,3}")
    if s[-1] != "}":
        raise RuntimeError(
            f"Default value of IntArrayRef is missing right brace '}}', found {s[-1]}"
        )

    inner = s[1:-1]  # strip { }
    if not inner.strip():
        return []

    args = []
    for tok in inner.split(","):
        tok = tok.strip()
        if tok:
            args.append(int(tok))
    return args

# source: https://github.com/pytorch/pytorch/blob/50eea6cd/torch/csrc/utils/python_arg_parser.cpp#L1339-L1398
def parse_string_literal(s: str) -> str:
    # C++: TORCH_CHECK(str.length() >= 2, "String defaults must be quoted");
    if len(s) < 2:
        raise RuntimeError("String defaults must be quoted")

    if s[0] == '"':
        if s[-1] != '"':
            raise RuntimeError(f"Mismatched quotes in string default: {s}")
    else:
        if not (s[0] == "'" and s[-1] == "'"):
            raise RuntimeError(f"Invalid quotes in string default: {s}")

    out = []
    i = 1
    end = len(s) - 1

    while i < end:
        c = s[i]

        # обычный символ
        if c != '\\':
            out.append(c)
            i += 1
            continue

        # C++: TORCH_CHECK(i < str.size() - 2, "String ends with escaped final quote")
        if i >= end - 1:
            raise RuntimeError(f"String ends with escaped final quote: {s}")

        esc = s[i + 1]
        # поддерживаемые escape-последовательности
        if esc in ('\\', "'", '"'): out.append(esc)
        elif esc == 'a':            out.append('\a')
        elif esc == 'b':            out.append('\b')
        elif esc == 'f':            out.append('\f')
        elif esc == 'n':            out.append('\n')
        elif esc == 'v':            out.append('\v')
        elif esc == 't':            out.append('\t')
        else:
            raise RuntimeError(f"Unsupported escape sequence in string default: \\{esc}")
        i += 2

    return "".join(out)

numpy_compatibility_arg_names = {
    "dim": ("axis",),
    "keepdim": ("keepdims",),
    "input": ("x", "a", "x1"), 
    "other": ("x2",),
}



class Parameter:
    # source: https://github.com/pytorch/pytorch/blob/50eea6cd/torch/csrc/utils/python_arg_parser.cpp#L129-L174
    def __init__(self, fmt: str, keyword_only: bool):
        self.keyword_only = keyword_only
        # self.allow_numbers_as_tensors = False # задаётся в Signature.__init__

        self.default_int = 0
        self.default_bool = False
        self.default_double = None
        self.default_complex = (0, 0)
        self.default_scalar = None
        self.default_intlist = None
        self.default_scalartype = None
        self.default_layout = None
        self.default_string = None

        bracket = fmt.find("(")
        if bracket != -1:
            # from: Tensor(a!, b!, c!)[]? maybe
            # to:   Tensor[]? maybe
            close_bracket = fmt.find(")", bracket)
            if close_bracket == -1:
                raise RuntimeError(f"FunctionParameter(): missing closing annotation parenthesis: {fmt}")
            fmt = f"{fmt[:bracket]}{fmt[close_bracket + 1:]}"

        # ---- C++: parse "type name[=default]" ----
        space = fmt.find(" ")
        if space == -1:
            raise RuntimeError(f"FunctionParameter(): missing type: {fmt}")

        type_str = fmt[:space]
        name_str = fmt[space + 1:]

        # ---- ? → allow_none ----
        q = type_str.find("?")
        if q != -1:
            self.allow_none = True
            type_str = type_str[:q]
        else:
            self.allow_none = False

        # ---- [] / [N] → size ----
        bracket = type_str.find("[")
        if bracket != -1:
            size_str = type_str[bracket + 1 : -1]
            self.size = int(size_str) if size_str else 0
            type_str = type_str[:bracket]
        else:
            self.size = 0

        # ---- map type ----
        if type_str not in TYPE_MAP:
            raise RuntimeError(f"FunctionParameter(): invalid type string: {type_str}")
        self.type = TYPE_MAP[type_str]

        # ---- name + default ----
        eq = name_str.find("=")
        if eq != -1:
            self.name = name_str[:eq]
            self.optional = True
            self.set_default_str(name_str[eq + 1:])
        else:
            self.name = name_str
            self.optional = False
            self.default_value = None

        # ---- python_name ----
        self.python_name = self.name

        # ---- numpy aliases ----
        self.numpy_python_names = numpy_compatibility_arg_names.get(self.name, [])

    # source: https://github.com/pytorch/pytorch/blob/50eea6cd/torch/csrc/utils/python_arg_parser.cpp#L1400-L1485
    def set_default_str(self, str):
        if str == "None":
            self.allow_none = True
        type_ = self.type
        if type_ == ParameterType.TENSOR or type_ == ParameterType.DISPATCH_KEY_SET:
            if str != "None":
                raise RuntimeError(f"default value for Tensor must be none, got: {str}")
        elif type_ == ParameterType.INT64 or type_ == ParameterType.SYM_INT:
            if str != "None":
                self.default_int = int(str)
        elif type_ == ParameterType.BOOL:
            self.default_bool = str == "True" or str == "true"
        elif type_ == ParameterType.DOUBLE:
            self.default_double = float(str)
        elif type_ == ParameterType.COMPLEX:
            self.default_complex = (float(str), 0) # TODO: parse "x + xj"?
            # ахах, в python это просто complex(str), но сделаю, как в реальном aten
        elif type_ == ParameterType.SCALAR:
            if str != "None":
                # we sometimes rely on integer-vs-float values, e.g. with arange.
                self.default_scalar = float(str) if '.' in str or 'e' in str or 'E' in str else int(str)
        elif type_ == ParameterType.INT_LIST or type_ == ParameterType.SYM_INT_LIST:
            if str != "None":
                self.default_intlist = parse_intlist_args(str, self.size)
        elif type_ == ParameterType.FLOAT_LIST:
            if str != "None":
                raise RuntimeError("Defaults not supported for float[]")
        elif type_ == ParameterType.SCALARTYPE:
            if str == "None":
                self.default_scalartype = None
            elif str == "torch.int64":
                self.default_scalartype = torch.int64
            else:
                raise RuntimeError(f"invalid default value for ScalarType: {str}")
        elif type_ == ParameterType.LAYOUT:
            if str == "None":
                pass
            elif str == "torch.strided":
                self.default_layout = torch.strided
            elif str == "torch.sparse_coo":
                self.default_layout = torch.sparse_coo
            else:
                raise RuntimeError(f"invalid default value for layout: {str}")
        elif type_ == ParameterType.DEVICE:
            if str != "None":
                raise RuntimeError(f"invalid device: {str}")
        elif type_ == ParameterType.STREAM:
            if str != "None":
                raise RuntimeError(f"invalid stream: {str}")
        elif type_ == ParameterType.STRING:
            if str != "None":
                self.default_string = parse_string_literal(str)
        # These types weren't handled here before. Adding a default error
        # led to a lot of test failures so adding this skip for now.
        # We should correctly handle these though because it might be causing
        # silent failures.
        elif type_ == ParameterType.TENSOR_LIST:
            pass # throw std::runtime_error("Invalid Tensor List")
        elif type_ == ParameterType.GENERATOR:
            pass # throw std::runtime_error("ParameterType::GENERATOR")
        elif type_ == ParameterType.PYOBJECT:
            pass # throw std::runtime_error("ParameterType::PYOBJECT")
        elif type_ == ParameterType.MEMORY_FORMAT:
            pass # throw std::runtime_error("ParameterType::MEMORY_FORMAT")
        elif type_ == ParameterType.DIMNAME:
            pass # throw std::runtime_error("ParameterType::DIMNAME");
        elif type_ == ParameterType.DIMNAME_LIST:
            pass # throw std::runtime_error("ParameterType::DIMNAME_LIST")
        elif type_ == ParameterType.SCALAR_LIST:
            pass # throw std::runtime_error("ParameterType::SCALAR_LIST")
        elif type_ == ParameterType.STORAGE:
            pass # throw std::runtime_error("ParameterType::STORAGE")
        elif type_ == ParameterType.QSCHEME:
            pass # throw std::runtime_error("ParameterType::QSCHEME")
        else:
            raise RuntimeError("unknown parameter type")
        self.default_value = str

    def check(self, obj, overloaded_args, argnum, failed_idx):
        # 1. Основная проверка типа
        if CHECK_TABLE[self.type](self, obj, overloaded_args, argnum, failed_idx):
            return True

        # 2. Если тип не подошёл, но есть __torch_function__ (и это не Tensor)
        # NB: PyTorch НЕ допускает Tensor subclasses здесь — они уже обработаны внутри check_tensor
        if has_torch_function_unary(obj) and not isinstance(obj, torch.Tensor):
            overloaded_args.append(obj)
            return True
        return False

    def type_name(self):
        # человекочитаемое имя для ошибок
        try:             return TYPE_NAME_MAP[self.type]
        except KeyError: raise RuntimeError(f"unknown parameter type: {self.type!r}")



_allowed = {
    "add", "add_", "add_out",
    "div", "div_", "div_out",
    "divide", "divide_", "divide_out",       # alias of div
    "mul", "mul_", "mul_out",
    "multiply", "multiply_", "multiply_out", # alias of mul
    "sub", "sub_", "sub_out",
    "subtract", "subtract_", "subtract_out", # alias of sub
    "true_divide", "true_divide_", "true_divide_out",
    "to", "_to_copy", "copy_", "copy",
    "floor_divide", "floor_divide_", "floor_divide_out",
    "_conj"
}
def should_allow_numbers_as_tensors(name):
    return name in _allowed

def find_matching_paren(s, start):
    # start — позиция после '('
    depth = 1
    i = start
    while i < len(s):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise RuntimeError(f"missing closing parenthesis in: {s}")

def find_comma_outside_parens(s, start):
    # Tensor(a!, b!) out, Tensor(a!, b!) other
    #                   ^ ище только это!
    depth = 0
    i = start
    while i < len(s):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
        elif s[i] == ',' and depth == 0:
            return i
        i += 1
    return -1

class Signature:
    # source: https://github.com/pytorch/pytorch/blob/50eea6cd/torch/csrc/utils/python_arg_parser.cpp#L1488-L1547
    def __init__(self, fmt: str, index: int = 0):
        self.index = index

        # --- 1. Найти '(' ---
        open_paren = fmt.find("(")
        if open_paren == -1:
            raise RuntimeError(f"missing opening parenthesis: {fmt}")

        self.name = fmt[:open_paren]

        # --- 2. allow_numbers_as_tensors ---
        allow_numbers_as_tensors = should_allow_numbers_as_tensors(self.name)

        # --- 3. Разбор параметров ---
        last_offset = open_paren + 1
        keyword_only = False
        done = False
        self.params = []

        while not done:
            offset = find_comma_outside_parens(fmt, last_offset)
            next_offset = offset + 2

            if offset == -1:
                offset = find_matching_paren(fmt, last_offset)
                done = True
                next_offset = offset + 1

                # пустой список параметров: fn()
                if offset == last_offset:
                    last_offset = next_offset
                    break

            if offset == -1:
                raise RuntimeError(f"missing closing parenthesis: {fmt}")
            if offset == last_offset:
                raise RuntimeError(f"malformed signature: {fmt}")

            param_str = fmt[last_offset:offset]
            last_offset = next_offset

            if param_str == "*":
                keyword_only = True
            else:
                p = Parameter(param_str, keyword_only)
                p.allow_numbers_as_tensors = allow_numbers_as_tensors
                self.params.append(p)

        # --- 4. |deprecated / |hidden ---
        self.hidden     = False
        self.deprecated = False

        tail = fmt[last_offset:]
        if tail == "|deprecated":
            self.hidden = True
            self.deprecated = True
        elif tail == "|hidden":
            self.hidden = True

        # --- 5. max_args ---
        self.max_args = len(self.params)

        # --- 6. min_args + max_pos_args ---
        self.min_args = 0
        self.max_pos_args = 0
        for p in self.params:
            if not p.optional:
                self.min_args += 1
            if not p.keyword_only:
                self.max_pos_args += 1

    # source: https://github.com/pytorch/pytorch/blob/50eea6cd/torch/csrc/utils/python_arg_parser.cpp#L1549-L1571
    def __repr__(self):
        parts = []
        keyword_already = False

        for i, param in enumerate(self.params):
            if i:
                parts.append(", ")

            if param.keyword_only and not keyword_already:
                parts.append("*, ")
                keyword_already = True

            parts.append(f"{param.type_name()} {param.name}")

            if param.optional:
                parts.append(f" = {param.default_value}")

        return f"({''.join(parts)})"

    # source: https://github.com/pytorch/pytorch/blob/50eea6cd/torch/csrc/utils/python_arg_parser.cpp#L1680-L1826
    def parse(self, self_obj, args, kwargs, overloaded_args, throw_error=False):
        """
        Полный Python-порт FunctionSignature::parse.
            self.params — список Parameter
            self.max_pos_args — максимум позиционных аргументов
            self.name — имя оператора
        """

        nargs = len(args)
        remaining_kwargs = len(kwargs) if kwargs else 0
        arg_pos = 0
        allow_varargs_intlist = False

        # --- 1. Проверка varargs IntArrayRef ---
        if self.max_pos_args == 1 and self.params[0].type in (ParameterType.INT_LIST, ParameterType.SYM_INT_LIST):
            failed_idx = [-1]
            allow_varargs_intlist = is_int_or_symint_list(self.params[0], args, None, 0, failed_idx)

        # --- 2. Проверка количества позиционных аргументов ---
        if nargs > self.max_pos_args and not allow_varargs_intlist:
            if throw_error:
                raise TypeError(
                    f"{self.name}() takes {self.max_pos_args} positional "
                    f"arguments but {nargs} were given"
                )
            return None

        dst = []

        # --- 3. torch_function hook ---
        if self_obj is not None and has_torch_function_unary(self_obj):
            overloaded_args.append(self_obj)

        # --- 4. Основной цикл по параметрам ---
        for i, param in enumerate(self.params):
            obj = None
            is_kwd = False

            # 4.1. Позиционный аргумент
            if arg_pos < nargs:
                if param.keyword_only:
                    if throw_error:
                        raise TypeError(
                            f"{self.name}() got extra positional arguments"
                        )
                    return None
                obj = args[arg_pos]

            # 4.2. Иначе ищем в kwargs
            elif kwargs:
                obj = kwargs.get(param.python_name)
                if obj is None:
                    # numpy-style алиасы
                    for np_name in param.numpy_python_names:
                        obj = kwargs.get(np_name)
                        if obj is not None:
                            break
                is_kwd = True

            # --- 5. Проверка optional / allow_none ---
            failed_idx = [-1]
            varargs_eligible = (
                allow_varargs_intlist and arg_pos == 0 and not is_kwd
            )

            if ((obj is None and param.optional) or
                (obj is None and param.allow_none)):
                dst.append(None)

                # --- 9. Сдвиг позиции ---
                if not is_kwd:
                    arg_pos += 1
                elif obj is not None:
                    remaining_kwargs -= 1
                continue

            if obj is None:
                if throw_error:
                    raise TypeError(
                        f"{self.name}() missing required argument '{param.name}'"
                    )
                return None

            # --- 6. Основная проверка типа ---
            if param.check(obj, overloaded_args, i, failed_idx):
                dst.append(obj)

                # --- 9. Сдвиг позиции ---
                if not is_kwd:
                    arg_pos += 1
                elif obj is not None:
                    remaining_kwargs -= 1
                continue

            # --- 7. varargs IntArrayRef ---
            if varargs_eligible and is_int_or_symint_list(param, args, None, 0, failed_idx):
                dst.append(args)
                arg_pos = nargs
                continue

            # --- 8. Ошибка типа ---
            if not throw_error:
                return None

            # element-level mismatch (list element)
            if failed_idx[0] != -1:
                bad = failed_idx[0]
                seq = obj if isinstance(obj, (list, tuple)) else args
                elem = seq[bad]
                raise TypeError(
                    f"{self.name}(): argument '{param.name}' "
                    f"(position {arg_pos+1}) must be {param.type_name()}, "
                    f"but element at pos {bad} is {type(elem).__name__}"
                )

            # normal mismatch
            if is_kwd:
                raise TypeError(
                    f"{self.name}(): argument '{param.name}' must be "
                    f"{param.type_name()}, not {type(obj).__name__}"
                )
            raise TypeError(
                f"{self.name}(): argument '{param.name}' "
                f"(position {arg_pos+1}) must be {param.type_name()}, "
                f"not {type(obj).__name__}"
            )

        # --- 10. Проверка лишних kwargs ---
        if remaining_kwargs > 0:
            if throw_error:
                extra = next(iter(kwargs.keys()))
                raise TypeError(
                    f"{self.name}() got an unexpected keyword argument '{extra}'"
                )
            return None

        return dst



from abc import ABC, abstractmethod
from typing import Any, List, Dict, Tuple
import sys

# source of all types: https://github.com/pytorch/pytorch/blob/2cb9a5bf/torch/csrc/utils/invalid_arguments.cpp#L19-L124

class Type(ABC):
    @abstractmethod
    def is_matching(self, obj: Any) -> bool:
        ...

class SimpleType(Type):
    def __init__(self, name: str) -> None:
        self.name = name

    def is_matching(self, obj: Any) -> bool:
        return type(obj).__name__ == self.name

class MultiType(Type):
    def __init__(self, accepted_types: List[str]) -> None:
        self.types = list(accepted_types)

    def is_matching(self, obj: Any) -> bool:
        return type(obj).__name__ in self.types

class NullableType(Type):
    def __init__(self, inner: Type) -> None:
        self.inner = inner

    def is_matching(self, obj: Any) -> bool:
        return obj is None or self.inner.is_matching(obj)

class TupleType(Type):
    def __init__(self, types: List[Type]) -> None:
        self.types = types

    def is_matching(self, obj: Any) -> bool:
        if not isinstance(obj, tuple):
            return False
        if len(obj) != len(self.types):
            return False
        return all(t.is_matching(o) for t, o in zip(self.types, obj))

class SequenceType(Type):
    def __init__(self, inner: Type) -> None:
        self.inner = inner

    def is_matching(self, obj: Any) -> bool:
        # максимально близко к PySequence_Check, но без C‑API
        if isinstance(obj, (str, bytes)):
            return False
        try:
            it = iter(obj)
        except TypeError:
            return False
        for x in it:
            if not self.inner.is_matching(x):
                return False
        return True



class Argument:
    def __init__(self, name: str, type_: Type) -> None:
        self.name = name
        self.type = type_

class Option:
    def __init__(self, arguments: List[Argument], is_variadic: bool, has_out: bool) -> None:
        self.arguments = arguments
        self.is_variadic = is_variadic
        self.has_out = has_out



def _split_string(s: str, sep: str = ", ") -> List[str]:
    return s.split(sep) if s else []

# source: https://github.com/pytorch/pytorch/blob/2cb9a5bf/torch/csrc/utils/invalid_arguments.cpp#L140-L163
def _build_type(type_name: str, is_nullable: bool) -> Type:
    # float → MultiType(float, int, long)
    if type_name == "float":
        base = MultiType(["float", "int", "long"])

    # int → MultiType(int, long)
    elif type_name == "int":
        base = MultiType(["int", "long"])

    # tuple[...] → TupleType
    elif type_name.startswith("tuple["):
        inner = type_name[len("tuple["):-1]  # remove "tuple[" and trailing "]"
        parts = _split_string(inner, ",")
        types = [_build_type(p, False) for p in parts]
        base = TupleType(types)

    # sequence[...] → SequenceType
    elif type_name.startswith("sequence["):
        inner = type_name[len("sequence["):-1]
        base = SequenceType(_build_type(inner, False))

    # simple type
    else:
        base = SimpleType(type_name)

    # nullable wrapper
    return NullableType(base) if is_nullable else base



# source: https://github.com/pytorch/pytorch/blob/2cb9a5bf/torch/csrc/utils/invalid_arguments.cpp#L165-L225
def parse_option(option_str: str,
                 kwargs: Dict[str, Any]) -> Tuple[Option, str]:
    if option_str == "no arguments":
        return Option([], False, False), option_str

    has_out = False
    arguments: List[Argument] = []
    printable_option = option_str
    inner = option_str[1:len(option_str) - 1]  # убираем внешние скобки

    # hack для out‑аргумента (маркер '#')
    out_pos = printable_option.find('#')
    if out_pos != -1:
        if "out" in kwargs:
            kwonly_part = printable_option[out_pos + 1:]
            printable_option = f"{printable_option[:out_pos]}*, {kwonly_part}"
        elif out_pos >= 2:
            printable_option = f"{printable_option[:out_pos - 2]})"
        else:
            printable_option = f"{printable_option[:out_pos]})"
        has_out = True

    for arg in _split_string(inner, ", "):
        is_nullable = False
        type_start_idx = 0

        if arg and arg[0] == '#':
            type_start_idx += 1

        if type_start_idx < len(arg) and arg[type_start_idx] == '[':
            is_nullable = True
            type_start_idx += 1
            suffix = " or None]"
            if arg.endswith(suffix):
                arg = arg[:len(arg) - len(suffix)]

        type_end_idx = arg.rfind(' ')
        name_start_idx = type_end_idx + 1

        dots_idx = arg.find("...")
        if dots_idx != -1:
            type_end_idx -= 4  # убираем " ..."

        type_name = arg[type_start_idx:type_end_idx]
        name = arg[name_start_idx:]

        arguments.append(Argument(name, _build_type(type_name, is_nullable)))

    is_variadic = "..." in inner
    return Option(arguments, is_variadic, has_out), printable_option

# source: https://github.com/pytorch/pytorch/blob/2cb9a5bf/torch/csrc/utils/invalid_arguments.cpp#L227-L238
def argcount_match(option, arguments, kwargs):
    num_expected = len(option.arguments)
    num_got = len(arguments) + len(kwargs)

    # out‑аргумент уменьшает ожидаемое число, если out не передан
    if option.has_out and "out" not in kwargs:
        num_expected -= 1

    return (
        num_got == num_expected
        or (option.is_variadic and num_got > num_expected)
    )

# source: https://github.com/pytorch/pytorch/blob/2cb9a5bf/torch/csrc/utils/invalid_arguments.cpp#L240-L318
# а вот идея с append уже моя
def formatted_arg_desc(option, arguments, kwargs, append):
    # ANSI‑цвета, как в C++
    if sys.stdout.isatty() and sys.stderr.isatty():
        RED         = "\33[31;1m"
        RESET_RED   = "\33[0m"
        GREEN       = "\33[32;1m"
        RESET_GREEN = "\33[0m"
    else:
        RED         = "!"
        RESET_RED   = "!"
        GREEN       = ""
        RESET_GREEN = ""

    num_args = len(arguments) + len(kwargs)
    append("(")

    for i in range(num_args):
        if i:
            append(", ")

        is_kwarg = i >= len(arguments)
        if is_kwarg:
            name = option.arguments[i].name
            arg = kwargs[name]
        else:
            arg = arguments[i]

        # matching logic
        if i < len(option.arguments):
            is_matching = option.arguments[i].type.is_matching(arg)
        elif option.is_variadic:
            is_matching = option.arguments[-1].type.is_matching(arg)
        else:
            is_matching = False

        # цвет
        append(GREEN if is_matching else RED)

        # kwarg prefix
        if is_kwarg:
            append(f"{option.arguments[i].name}=")

        # tuple/list pretty‑print
        if isinstance(arg, (tuple, list)):
            append(type(arg).__name__)
            append(" of ")

            append("(" if isinstance(arg, tuple) else "[")

            for j, elem in enumerate(arg):
                if j:
                    append(", ")
                append(type(elem).__name__)

            if isinstance(arg, tuple):
                if len(arg) == 1:
                    append(",")
                append(")")
            else:
                append("]")

        else:
            append(type(arg).__name__)

        # reset color
        append(RESET_GREEN if is_matching else RESET_RED)

    append(")")

# source: https://github.com/pytorch/pytorch/blob/2cb9a5bf/torch/csrc/utils/invalid_arguments.cpp#L320-L332
# а вот идея с append уже моя
def arg_desc(arguments, kwargs, append):
    append("(")
    first = True

    for arg in arguments:
        if first:
            first = False
        else:
            append(", ")
        append(type(arg).__name__)

    for name, value in kwargs.items():
        if first:
            first = False
        else:
            append(", ")
        append(f"{name}={type(value).__name__}")

    append(")")

# source: https://github.com/pytorch/pytorch/blob/2cb9a5bf/torch/csrc/utils/invalid_arguments.cpp#L334-L356
def try_match_kwargs(option: Option,
                     kwargs: Dict[str, Any]) -> List[str]:
    unmatched: List[str] = []

    start_idx = len(option.arguments) - len(kwargs)
    if option.has_out and "out" not in kwargs:
        start_idx -= 1
    if start_idx < 0:
        start_idx = 0

    for key in kwargs.keys():
        found = False
        for i in range(start_idx, len(option.arguments)):
            if option.arguments[i].name == key:
                found = True
                break
        if not found:
            unmatched.append(key)

    return unmatched



# source: https://github.com/pytorch/pytorch/blob/2cb9a5bf/torch/csrc/utils/invalid_arguments.cpp#L360-L444
# а вот идея с append уже моя
def format_invalid_args(given_args, given_kwargs, function_name, options):
    args = list(given_args)
    kwargs = dict(given_kwargs or {})

    msg = [f"{function_name} received an invalid combination of arguments - "]
    append = msg.append

    if len(options) == 1:
        option, option_str = parse_option(options[0], kwargs)
        unmatched = try_match_kwargs(option, kwargs)

        if unmatched:
            append("got unrecognized keyword arguments: ")
            append(", ".join(unmatched))
        else:
            append("got ")
            if argcount_match(option, args, kwargs):
                formatted_arg_desc(option, args, kwargs, append)
            else:
                arg_desc(args, kwargs, append)
            append(f", but expected {option_str}")

        return ''.join(msg)

    # multiple overloads
    append("got ")
    arg_desc(args, kwargs, append)
    append(", but expected one of:\n")

    for opt_str in options:
        option, printable = parse_option(opt_str, kwargs)
        append(f" * {printable}\n")

        if argcount_match(option, args, kwargs):
            unmatched = try_match_kwargs(option, kwargs)
            if unmatched:
                append("      didn't match because some of the keywords were incorrect: ")
                append(", ".join(unmatched))
                append("\n")
            else:
                append("      didn't match because some of the arguments have invalid types: ")
                formatted_arg_desc(option, args, kwargs, append)
                append("\n")

    return ''.join(msg)



class PythonArgs:
    def __init__(self, traceable, signature, overloaded_args):
        self.traceable = traceable
        self.signature = signature
        self.overloaded_args = overloaded_args

class PythonArgParser:
    # source: https://github.com/pytorch/pytorch/blob/50eea6cd/torch/csrc/utils/python_arg_parser.cpp#L1828-L1851
    def __init__(self, *fmts, traceable=False):
        self.traceable = traceable
        self.signatures = [Signature(fmt, i) for i, fmt in enumerate(fmts)]
        self.max_args = max(sig.max_args for sig in self.signatures)
        self.function_name = self.signatures[0].name.split(".", 1)[0] if self.signatures else None

        # deprecated → в конец
        self.signatures.sort(key=lambda s: s.deprecated)

    # source: https://github.com/pytorch/pytorch/blob/50eea6cd/torch/csrc/utils/python_arg_parser.cpp#L1853-L1872
    def check_deprecated(self, signature):
        if signature.deprecated:
            msg = f"This overload of {signature.name} is deprecated:\n\t{signature.name}{signature.to_string()}"
            options = self.get_signatures()
            if options:
                msg += "\nConsider using one of the following signatures instead:"
                for sig in options:
                    msg += f"\n\t{signature.name}{sig}"
            print("[ATEN WARNING]", msg)  # or warn_once

    # source: https://github.com/pytorch/pytorch/blob/50eea6cd/torch/csrc/utils/python_arg_parser.cpp#L1874-L1899
    def raw_parse(self, self_obj, args, kwargs):
        # 1. Если перегрузка одна — сразу пробуем её жёстко
        if len(self.signatures) == 1:
            sig = self.signatures[0]
            overloaded_args = []
            # throw_error=True → если что-то не так, parse сам бросит исключение
            sig.parse(self_obj, args, kwargs, overloaded_args, throw_error=True)
            self.check_deprecated(sig)
            return PythonArgs(self.traceable, sig, overloaded_args)

        # 2. Иначе — мягкий проход по всем сигнатурам
        for sig in self.signatures:
            overloaded_args = []
            ok = sig.parse(self_obj, args, kwargs, overloaded_args, throw_error=False)
            if ok:
                self.check_deprecated(sig)
                return PythonArgs(self.traceable, sig, overloaded_args)

        # 3. Если ни одна не подошла — формируем красивую ошибку
        self.print_error(self_obj, args, kwargs)

    # source: https://github.com/pytorch/pytorch/blob/50eea6cd/torch/csrc/utils/python_arg_parser.cpp#L1901-L1928
    def print_error(self, self_obj, args, kwargs):
        # 1. Посчитать количество аргументов
        num_args = len(args) + (len(kwargs) if kwargs else 0)

        # 2. Найти "правдоподобные" сигнатуры по количеству аргументов
        plausible = []
        for i, sig in enumerate(self.signatures):
            if sig.min_args <= num_args <= sig.max_args and not sig.hidden:
                plausible.append(i)

        # 3. Если ровно одна — пробуем её распарсить "жёстко"
        if len(plausible) == 1:
            sig = self.signatures[plausible[0]]
            # В C++ здесь вызывается signature.parse(..., throw_error=True)
            # В Python мы просто вызываем parse() и позволяем ему бросить исключение
            sig.parse(self_obj, args, kwargs, throw_error=True)

        # 4. Если мы здесь — значит parse() не бросил, но сигнатура не подошла
        options = self.get_signatures()
        # Ради только этого огромный подмодуль сверху :)
        # Целая поисковая машина для красивого вывода ошибки!)
        msg = format_invalid_args(args, kwargs, f"{self.function_name}()", options)

        raise TypeError(msg)

    # source: https://github.com/pytorch/pytorch/blob/50eea6cd/torch/csrc/utils/python_arg_parser.cpp#L1930-L1938
    def get_signatures(self):
        return [repr(sig) for sig in self.signatures if not sig.hidden]



# https://github.com/pytorch/pytorch/blob/2cb9a5bf/torch/csrc/Event.cpp#L26-L28
Event_parser = PythonArgParser(
    "Event(Device device=None, *, bool enable_timing=False, bool blocking=False, bool interprocess=False)",
)

# https://github.com/pytorch/pytorch/blob/2cb9a5bf/torch/csrc/Event.cpp#L134-L136
from_ipc_handle_parser = PythonArgParser(
    "from_ipc_handle(Device device, std::string ipc_handle)",
)

# https://github.com/pytorch/pytorch/blob/2cb9a5bf/torch/csrc/autograd/python_nested_functions_manual.cpp#L14-L16
nested_tensor_parser = PythonArgParser(
    "nested_tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None, bool pin_memory=False, bool requires_grad=False)",
)

# https://github.com/pytorch/pytorch/blob/2cb9a5bf/torch/csrc/Device.cpp#L51-L53
device_parser = PythonArgParser(
    "device(Device device)",
    "device(std::string_view type, int64_t? index=-1)",
)

# https://github.com/pytorch/pytorch/blob/2cb9a5bf/tools/autograd/templates/python_nn_functions.cpp#L39-L43
to_parser = PythonArgParser(
    "to(Device device=None, ScalarType dtype=None, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
    "to(ScalarType dtype, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
    "to(Tensor tensor, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
)

# https://github.com/pytorch/pytorch/blob/2cb9a5bf/torch/csrc/autograd/init.cpp#L703-L705
set_autocast_enabled_parser = PythonArgParser(
    "set_autocast_enabled(std::string_view device_type, bool enabled)",
    "set_autocast_enabled(bool enabled)",
) # this signature is deprecated.

# https://github.com/pytorch/pytorch/blob/2cb9a5bf/torch/csrc/autograd/init.cpp#L726-L728
is_autocast_enabled_parser = PythonArgParser(
    "is_autocast_enabled(std::string_view device_type)",
    "is_autocast_enabled()",
) # this signature is deprecated.

# https://github.com/pytorch/pytorch/blob/2cb9a5bf/torch/csrc/utils/tensor_new.cpp#L611-L617
new_parser = PythonArgParser(
    "new(*, Device? device=None)",
    "new(*, int64_t cdata)|hidden",
    "new(Tensor indices, Tensor values, *, Device? device=None)",
    "new(Tensor indices, Tensor values, IntArrayRef size, *, Device? device=None)",
    "new(SymIntArrayRef size, *, Device? device=None)",
)

# https://github.com/pytorch/pytorch/blob/2cb9a5bf/torch/csrc/utils/tensor_new.cpp#L717-L730
legacy_new = PythonArgParser(
    "new(*, Device? device=None)",
    "new(Storage storage)",
    "new(*, int64_t cdata)|hidden",
    # This constructor is no longer legacy, it will also be usable for
    # subclass initialization
    "new(Tensor other)",
    "new(Tensor other, *, Device? device=None)|hidden", # prevent Tensor
                                                        # matching with
                                                        # IntArrayRef,
                                                        # PyObject*
    "new(SymIntArrayRef size, *, Device? device=None)",
    "new(PyObject* data, *, Device? device=None)",
)

# https://github.com/pytorch/pytorch/blob/2cb9a5bf/torch/csrc/TypeInfo.cpp#L41
finfo_parser = PythonArgParser(
    "finfo(ScalarType type)",
    "finfo()",
)

# https://github.com/pytorch/pytorch/blob/2cb9a5bf/torch/csrc/TypeInfo.cpp#L72-L74
iinfo_parser = PythonArgParser(
    "iinfo(ScalarType type)",
)

THPStorageStr = "torch.UntypedStorage"
THPStorage_parser = PythonArgParser(
    f"{THPStorageStr}(*, int64_t allocator=None, Device device=None)",
    f"{THPStorageStr}(int64_t size, *, int64_t allocator=None, Device device=None)",
    f"{THPStorageStr}(PyObject* sequence, *, int64_t allocator=None, Device device=None)",
)

# ... Чёт всё не то выпадает при поиске PythonArgParser в github



#     Источник НУЖНЫХ нам функций:
# https://github.com/pytorch/pytorch/blob/2cb9a5bf/aten/src/ATen/native/native_functions.yaml

#     Самое главное, отсутствие которого было бы фатально для всего моего скрипта not_aten.py, это перегрузки вида:
# eye(SymInt n, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
# eye.m(SymInt n, SymInt m, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
# eye.out(SymInt n, *, Tensor(a!) out) -> Tensor(a!)
# eye.m_out(SymInt n, SymInt m, *, Tensor(a!) out) -> Tensor(a!)

import yaml
from collections import defaultdict

def load_native_functions(yaml_text):
    data   = yaml.safe_load(yaml_text)
    groups = defaultdict(list)

    for row in data:
        func = row["func"]
        name = func.split("(", 1)[0]
        base = name.split(".", 1)[0]
        groups[base].append(func)

    return {base: PythonArgParser(*group) for base, group in groups.items()}

native_functions = """
- func: eye(SymInt n, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  dispatch:
    CompositeExplicitAutograd: eye

- func: eye.m(SymInt n, SymInt m, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  dispatch:
    CompositeExplicitAutograd: eye

- func: eye.out(SymInt n, *, Tensor(a!) out) -> Tensor(a!)
  dispatch:
    CPU, Meta: eye_out_cpu
    CUDA: eye_out_cuda
    MPS: eye_out_mps

- func: eye.m_out(SymInt n, SymInt m, *, Tensor(a!) out) -> Tensor(a!)
  dispatch:
    CPU, Meta: eye_out_cpu
    CUDA: eye_out_cuda
    MPS: eye_out_mps
"""

loaded_natives = load_native_functions(native_functions)

if __name__ == "__main__":
    # for sig in loaded_natives["eye"].signatures:
    #     print(sig.name, sig)

    for kwargs in ({}, {"out": torch.randn(2, 3)}):
        for args in ((1,), (True, 2), (1.0, 2)):
            try:
                py_args = loaded_natives["eye"].raw_parse(None, args, kwargs)
                print(py_args.signature.name, py_args.overloaded_args)
            except TypeError as e:
                print(e) # с первого раза получил корректный вывод ошибок :)))

""" Проверочный вывод:
eye []
eye.m []
eye() received an invalid combination of arguments - got (float, int), but expected one of:
 * (int n, *, torch.dtype dtype = None, torch.layout layout = None, torch.device device = None, bool pin_memory = None)
 * (int n, int m, *, torch.dtype dtype = None, torch.layout layout = None, torch.device device = None, bool pin_memory = None)
 * (int n, *, Tensor out)
 * (int n, int m, *, Tensor out)

eye.out []
eye.m_out []
eye() received an invalid combination of arguments - got (float, int, out=Tensor), but expected one of:
 * (int n, *, torch.dtype dtype = None, torch.layout layout = None, torch.device device = None, bool pin_memory = None)
 * (int n, int m, *, torch.dtype dtype = None, torch.layout layout = None, torch.device device = None, bool pin_memory = None)
 * (int n, *, Tensor out)
      didn't match because some of the arguments have invalid types: (float, int, out=Tensor)   float и int - красные, out=Tensor - зелёный!
 * (int n, int m, *, Tensor out)
"""
