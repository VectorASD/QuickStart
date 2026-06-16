from functools import lru_cache
from pathlib import Path
import regex
import re
from io import StringIO


workdir = Path(__file__).resolve().parent
source = workdir / "not_opapi.cpp"
dest   = workdir / "not_opapi_gen.cpp"

assert source.exists()


MAKE_OP_RE = regex.compile(
    r"""
    MAKE_OP\s*\(\s*
    (?P<name>\w+)                              # имя операции
    \s*
    (?P<signature>                              # сигнатура с внешними круглыми скобками
        \(
        (?: [^()] | (?&signature) )*            # не-скобки или рекурсивно сама себя
        \)
    )
    \s*
    (?P<body>                                   # тело с внешними фигурными скобками
        \{
        (?: [^{}] | (?&body) )*                 # не-скобки или рекурсивно сама себя
        \}
    )
    \s*\)
    """,
    regex.VERBOSE | regex.DOTALL
)

SIGNATURE_RE = re.compile(r'(?:const\s*)?([a-zA-Z_]\w*(?:\s+\*+|\**)|,)')

known_types = {
    "aclTensor*":      "t",
    "aclTensorList*":  "Lt",
    "aclScalar*":      "s",
    "float":           "f",
    "double":          "d",
    "int64_t":         "i",
    "bool":            "b",
    "uint64_t*":       "i*",
    "aclOpExecutor**": "E",
}

def parse_signature(signature: str):
  # print("|", signature)
  # print(SIGNATURE_RE.findall(signature))

    it = SIGNATURE_RE.finditer(signature)
    result = []
    simple_s = []
    for match in it:
        _type = match.group(1)
        if _type == ',':
            continue

        if _type == "sync":
            is_out = "sync"
            _type = next(it).group(1)
            assert _type == "aclTensor*"
        else:
            is_out = _type == "out"
            if is_out:
                _type = next(it).group(1)
        name = next(it).group(1)

        _type = _type.replace(' ', '')
        if is_out == "sync":
            simple = "$t"
        else:
            try: simple = known_types[_type]
            except KeyError:
                raise RuntimeError("Неизвестный тип:", _type) from None
            if is_out:
                simple = simple.upper()
        simple_s.append(simple)

        result.append((is_out, _type, name))

    return result, ''.join(simple_s)


def make_printer(func_name: str, need_out: bool, signature, write):
    tensors = []
    tensor_lists = []
    common = []
    for is_out, _type, name in signature:
        if bool(is_out) == need_out:
            if _type == "aclTensor*":
                tensors.append(name)
            elif _type == "aclTensorList*":
                tensor_lists.append(name)
            else:
                common.append((_type, name))

    write(f"\n    void {func_name}(")
    if not need_out:
        write("const char* opName, ")
    write("std::ostringstream& log) const {")

    if not need_out or common or tensors:
        write("\n        log")
    if not need_out:
        write(' << "[EXEC] " << opName')

    first = True
    for _type, name in common:
        write(' << "')
        if first:
            write(':')
            first = False
        if _type == "float":
            write(f' {name}=" << formatFloat({name})')
        elif _type == "double":
            write(f' {name}=" << formatDouble({name})')
        elif _type == "int64_t":
            write(f' {name}=" << {name}')
        elif _type == "bool":
            write(f' {name}=" << ({name} ? "true" : "false")')
        elif _type == "aclScalar*":
            write(f' {name}=" << {name}->toString()')
        else:
            raise RuntimeError(f"unknown printer type: {_type!r}")
    if not need_out and first and tensors:
        write(" << ':'")
        first = False

    for name in tensors:
        if first:
            first = False
        else:
            write("\n           ")
        write(rf' << "\n    {name}:\n" << tensorDataToString({name})')

    write(';');

    for name in tensor_lists:
        write(f"\n       aclTensorList::toString({name}, log);")
    write("\n    }")

exe_cache = {}
custon_names = {
    "tTi*E|x,out": "Unary",
}

def make_executor(signature, simple: str) -> str:
    key = f"{simple}|{','.join(rec[2] for rec in signature)}"
    try:
        return exe_cache[key][1]
    except KeyError: pass

    try:
        exe_name = f"{custon_names[key]}Executor"
    except KeyError:
        exe_name = f"Executor{len(exe_cache)}"

    output = StringIO()
    write = output.write

    write(f"struct {exe_name} {{")
    for is_out, _type, name in signature:
        is_const = "/*sync*/" if is_out == "sync" else "const"
        write(f"\n    {is_const} {_type} {name};")
    write('\n')
    make_printer("start", False, signature, write)
    make_printer("end",   True,  signature, write)
    write("\n};")

    exe_cache[key] = output.getvalue(), exe_name
    return exe_name


@lru_cache(maxsize=None)
def compile_SS_pattern(scalar_names: tuple[str]):
    pattern = rf"\b({'|'.join(map(re.escape, scalar_names))})\b"
    return re.compile(pattern).sub

def substitute_scalars(body: str, scalar_names: list[str]) -> str:
    if not scalar_names:
        return body
    names = tuple(sorted(scalar_names))
    return compile_SS_pattern(names)(r'exec->\1', body)

def make_GWS(op_name: str, exe_name: str, signature, body: str, write):
    write('\n\n\n__attribute__((visibility("default")))')
    write(f"\naclnnStatus {op_name}GetWorkspaceSize(")
    write(", ".join((
        *(
            f"{'/*sync*/' if is_out == 'sync' else 'const'} {_type} {name}"
            for is_out, _type, name in signature
        ),
        "uint64_t* workspaceSize", "aclOpExecutor** executor"
    )))
    write(") {")

    write("\n    ASSERT_CODE(workspaceSize && executor, INVALID_PARAM)")
    pointers = tuple(name for is_out, _type, name in signature if _type.endswith('*'))
    if pointers:
        write(f"\n    ASSERT({' && '.join(pointers)})")

    names = (rec[2] for rec in signature)
    write(f"\n    {exe_name}* exec = new {exe_name}{{{', '.join(names)}}};")
    write("\n    *workspaceSize = 0;")
    write("\n    *executor = reinterpret_cast<aclOpExecutor*>(exec);")
    write("\n    return OK;")
    write("\n}")

    write(f"\nDEFINE_ACLNN_OP({op_name}, {exe_name}, {{")
    tensors = tuple(name for is_out, _type, name in signature if _type == "aclTensor*")
    if tensors:
        write(f"\n    at::Tensor {', '.join(tensors)};")
        for name in tensors:
            write(f"\n    LOAD_TENSOR({name}, exec->{name});")

    scalar_names = []
    for is_out, _type, name in signature:
        if _type == "aclTensor*":
            pass
        elif _type == "aclScalar*":
            write(f"\n    const at::Tensor& {name} = exec->{name}->tensor;")
        elif _type == "aclTensorList*":
            write(f"\n    const at::TensorList& {name} = exec->{name}->aten_tensors();")
        elif _type in ("float", "double", "int64_t", "bool"):
            scalar_names.append(name)
        else:
            raise RuntimeError(f"unknown user type: {_type!r}")

    body = substitute_scalars(body, scalar_names)

    write("\n\n    ")
    write(body)

    first = True
    for is_out, _type, name in signature:
        if is_out == "sync":
            assert _type == "aclTensor*"
            if first:
                write('\n')
                first = False
            write(f"\n    SYNC_AFTER_MUTATION(exec->{name}, {name})")

    write("\n})")


def generate(log = False):
    result = StringIO()
    write = result.write

    content = source.read_text()
    for m in MAKE_OP_RE.finditer(content):
        op_name = m.group("name")
        raw_signature = m.group("signature").strip()[1:-1]  # убираем '(' и ')'
        body = m.group("body").strip()[1:-1].strip()  # убираем '{' и '}'
        signature, simple = parse_signature(raw_signature)

        assert signature and signature.pop() == (False, 'aclOpExecutor**', 'executor')
        assert signature and signature.pop() == (False, 'uint64_t*', 'workspaceSize')

        if log:
            print(op_name, "->", signature, f"({simple})", "->", body)
        exe_name = make_executor(signature, simple)

        make_GWS(op_name, exe_name, signature, body, write)

    with open(dest, "w") as file:
        file.write('#include "not_opapi_base.h"')
        for exe, exe_name in exe_cache.values():
            file.write("\n\n")
            file.write(exe)
        file.write(result.getvalue())
        file.write('\n')

if __name__ == "__main__":
    generate()
