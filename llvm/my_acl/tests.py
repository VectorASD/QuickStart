import torch

def tensor_str(t):
    if isinstance(t, tuple):
        return "(" + ", ".join(tensor_str(x) for x in t) + ")"
    # Убираем лишние пробелы и переводы строк для компактности
    return " ".join(line.strip() for line in str(t).split("\n"))

def _try_reduce_call(func, args_list, tensor_cpu=None, func_cpu=None):
    for args, kwargs in args_list:
        try:
            res = func(*args, **kwargs)
            res_str = tensor_str(res)
        except Exception as e:
            res_str = f"ERROR {type(e).__name__}: {e}"

        dim_val = kwargs.get('dim', None)
        keepdim = kwargs.get('keepdim', False)
        prefix = f"    dim={dim_val}, keepdim={keepdim}: "
        print(f"{prefix}{res_str}")

        if tensor_cpu is not None:
            try:
                cpu_func = func_cpu if func_cpu else getattr(tensor_cpu, func.__name__)
                cpu_res = cpu_func(*args, **kwargs)
                cpu_str = tensor_str(cpu_res)
            except Exception as e:
                cpu_str = f"ERROR {type(e).__name__}: {e}"
            expected_prefix = "    expected: "
            padding = " " * (len(prefix) - len(expected_prefix))
            print(f"{padding}{expected_prefix}{cpu_str}")

def _build_tensor_args(op_name):
    """Аргументы для методов тензора – dim всегда как keyword."""
    args_list = []
    if op_name in ("max", "min"):
        for dim_val in (0, 1):
            for keepdim in (False, True):
                args_list.append(((), {"dim": dim_val, "keepdim": keepdim}))
    else:
        for dim_val in (0, 1):
            for keepdim in (False, True):
                args_list.append(((), {"dim": dim_val, "keepdim": keepdim}))
        if op_name in ("all", "any", "amax", "amin"):
            args_list.append(((), {"keepdim": False}))
            args_list.append(((), {"keepdim": True}))
    return args_list

def _build_torch_args(op_name, tensor):
    args_list = []
    no_dim_none = {"mean", "sum", "prod", "var", "std", "logsumexp"}
    for dim_val in (0, 1, None):
        if dim_val is None and op_name in no_dim_none:
            continue
        for keepdim in (False, True):
            if dim_val is None:
                args_list.append(((tensor,), {"keepdim": keepdim}))
            else:
                args_list.append(((tensor,), {"dim": dim_val, "keepdim": keepdim}))
    return args_list

def test_reduce_op(tensor, tensor_cpu, op_name):
    print(f"\n--- {op_name} ---")

    op = getattr(tensor, op_name, None)
    if op is not None:
        try:
            res = op()
            print(f"  (no args): {res}")
            cpu_op = getattr(tensor_cpu, op_name, None)
            if cpu_op:
                expected_res = cpu_op()
                print(f"           : {expected_res}")
            else:
                print("             (no cpu method)")
        except Exception as e:
            print(f"  (no args): ERROR {type(e).__name__}: {e}")

        args_list = _build_tensor_args(op_name)
        _try_reduce_call(op, args_list, tensor_cpu)
        return

    torch_op = getattr(torch, op_name, None)
    if torch_op is None:
        print(f"  {op_name} not found in tensor methods or torch functions")
        return

    print("  (found in torch, testing with dim)")
    args_list = _build_torch_args(op_name, tensor)
    _try_reduce_call(torch_op, args_list, tensor_cpu, func_cpu=torch_op)


if __name__ == "__main__":
    import os

    os.environ["NOT_NPU_QUIET"] = '1'
    reduce_ops = [
        "all", "any", "logsumexp",
        "max", "amax", "min", "amin",
        "mean", "prod", "sum",
        "var", "std", "std_mean"
    ]

  # del os.environ["NOT_NPU_QUIET"]
  # reduce_ops = ["var"]

    if not torch.npu.is_available():
        raise RuntimeError("NPU device required")

    torch.manual_seed(42)
    t = torch.randn(2, 3, device='npu')
    t_cpu = t.cpu()
    print("Original tensor:\n", t)

    for op_name in reduce_ops:
        test_reduce_op(t, t_cpu, op_name)
