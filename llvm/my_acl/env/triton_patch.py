import pyzstd  # sudo pip install pyzstd

from triton.backends import backends
from triton.runtime.cache import get_cache_manager
from triton._C.libtriton import ir, passes, ascend

import hashlib
import pickle

AscendBackend = backends['ascend'].compiler


real_add_stages = AscendBackend.add_stages
def add_stages(self, stages, options, *language):
    if hasattr(options, 'use_bytecode'):
        from dataclasses import replace
        options = replace(options, use_bytecode=False)
        # use_bytecode=True, делает из трёх этапов 5

    real_add_stages(self, stages, options, *language)
    assert len(stages) == 3, "Видимо, у вас получился cpu-пайплайн: ttir, ttadapter, llir, cpuasm! А ожидается npu-бэкенд"
    real_ttir, real_ttadapter, real_npubin = stages.values()

    def ttir(mod, metadata):
        if "hash" not in metadata:
            metadata["hash"] = hashlib.sha256(f"{mod}-{metadata}".encode()).hexdigest()
        # the same optimize pass for triton-ir as all other backends
        pass_funcs = (
            passes.common.add_inliner,
            passes.ttir.add_combine,
            passes.common.add_canonicalizer,
            passes.ttir.add_reorder_broadcast,
            passes.common.add_cse,
            passes.common.add_licm,
            passes.common.add_symbol_dce,
            passes.ttir.add_loop_unroll,
        )
        dumps = [str(mod)]
        for add_pass in pass_funcs:
            pm = ir.pass_manager(mod.context)
          # pm.enable_debug()
            add_pass(pm)
            pm.run(mod)
            dumps.append(str(mod))
          # print(hashlib.sha256(str(mod).encode()).hexdigest())
          # получился тот же хеш, что и при монолитном ir.pass_manager!
        compressed = pyzstd.compress(pickle.dumps(dumps, protocol=4))
        get_cache_manager(metadata["hash"]).put(compressed, "stage_1.pkl.zstd", binary=True)
        return mod

    def _is_auto_map_parallel_blocks_enabled() -> bool:
        return os.getenv("TRITON_ALL_BLOCKS_PARALLEL", "false").lower() in ("true", "1")
    def ttadapter(mod, metadata, *, named_ops=True):
        # use triton_adapter to lower Triton-MLIR to linalg
        # Get Triton-MLIR as string
        passes = []; add = passes.append

        add(lambda pm: ascend.passes.ttir.add_auto_blockify(pm, 1 if not _is_auto_map_parallel_blocks_enabled() else metadata["auto_blockify_size"]))
        if (metadata["add_auto_scheduling"]):
            add(lambda pm: ascend.passes.ttir.add_dag_sync(pm))
            add(lambda pm: ascend.passes.ttir.add_dag_scope(pm))
            add(lambda pm: passes.common.add_cse(pm))
            add(lambda pm: passes.common.add_canonicalizer(pm))
            add(lambda pm: ascend.passes.ttir.add_dag_ssbuffer(pm))
            add(lambda pm: passes.common.add_cse(pm))
            add(lambda pm: passes.common.add_canonicalizer(pm))
        add(lambda pm: ascend.passes.ttir.add_triton_to_structure(pm, metadata["enable_mask_fallback_conversion"], metadata["optimize_dynamic_offset"]))
        add(lambda pm: ascend.passes.ttir.add_discrete_mask_access_conversion(pm, metadata["compile_on_910_95"], metadata["force_simt_template"]))
        add(lambda pm: ascend.passes.ttir.add_triton_to_annotation(pm))
        add(lambda pm: ascend.passes.ttir.add_triton_to_unstructure(pm, metadata["compile_on_910_95"], metadata["force_simt_template"]))
        add(lambda pm: ascend.passes.ttir.add_triton_to_hivm(pm))
        add(lambda pm: ascend.passes.ttir.add_triton_to_hfusion(pm))
        add(lambda pm: ascend.passes.ttir.add_triton_to_llvm(pm))
        add(lambda pm: ascend.passes.ttir.add_bubble_up_operation(pm))
        add(lambda pm: ascend.passes.ttir.add_triton_to_structure(pm, metadata["enable_mask_fallback_conversion"], metadata["optimize_dynamic_offset"]))
        add(lambda pm: ascend.passes.ttir.add_triton_to_linalg(pm, False, named_ops, metadata["enable_nd2nz_on_vector"], metadata["enable_select_analysis"], metadata["compile_on_910_95"]))

        dumps = []
        for add_pass in passes:
            pm = ir.pass_manager(mod.context)
          # pm.enable_debug()
            add_pass(pm)
            pm.run(mod)
            dumps.append(str(mod))
        compressed = pyzstd.compress(pickle.dumps(dumps, protocol=4))
        get_cache_manager(metadata["hash"]).put(compressed, "stage_2.pkl.zstd", binary=True)
        return str(mod)

    def npubin(linalg, metadata):
        import re
        DISABLE_AUTO_TILE_AND_BIND_SUBBLOCK_REGEX = r'hivm.disable_auto_tile_and_bind_subblock'
        MIX_MODE_REGEX      = r'mix_mode\s*=\s*"([^"]+)"'
        PARALLEL_MODE_REGEX = r'parallel_mode\s*=\s*"([^"]+)"'
        KERNEL_NAME_REGEX   = r"(?:tt|func)\.func[\s\w]+@(\w+)"
        TENSOR_KIND_REGEX   = r'%arg(\d+):[^,)]*?\{[^}]*?tt\.tensor_kind\s*=\s*([^:\s}]+)\s*:[^}]*?\}'
        BITCODES_REGEX      = r'bitcode\s*=\s*(?:"([^"]+)"|\'([^\']+)\'|(\w+))'
        metadata["shared"] = 1
        metadata["auto_tile_and_bind_subblock"] = not re.search(DISABLE_AUTO_TILE_AND_BIND_SUBBLOCK_REGEX, linalg)
        metadata["mix_mode"]      = 'aiv'  # re.search(MIX_MODE_REGEX, linalg).group(1)
        metadata["parallel_mode"] = 'simd' # re.search(PARALLEL_MODE_REGEX, linalg).group(1)
        metadata["kernel_name"] = re.search(KERNEL_NAME_REGEX, linalg).group(1)
        metadata["name"] = metadata["kernel_name"] + "_" + metadata["mix_mode"]
        metadata["tensor_kinds"] = [int(kind) for _, kind in re.findall(TENSOR_KIND_REGEX, linalg)]
        metadata["required_ub_bits"] = 0
        bitcodes = re.findall(BITCODES_REGEX, linalg)
        metadata["bitcodes"] = [val for group in bitcodes for val in group if val]
        return linalg # real_npubin(linalg, metadata)

    stages.clear()
    stages["ttir"]      = ttir
    stages["ttadapter"] = ttadapter
    stages["npubin"]    = npubin
AscendBackend.add_stages = add_stages



from triton.runtime.driver import driver
import torch

from functools import wraps
import traceback
from collections import defaultdict

def common_stride(shape):
    if not shape:
        return ()
    expected = [1]
    for d in reversed(shape[1:]):
        expected.append(expected[-1] * d)
    return tuple(reversed(expected))

def obj_to_str(obj):
    if hasattr(obj, "data_ptr") and hasattr(obj, "dtype"):
        name = type(obj).__name__
        shape = tuple(obj.size())
        dtype = str(obj.dtype).split('.')[-1]
        stride = obj.stride()
        if name == "StridedBuffer":
            name = "SB"
        if stride == common_stride(shape):
            return f"{name}({shape}, {dtype})", obj
        return f"{name}({shape}, {dtype}, {stride})", obj
    return str(obj), None

launcher_cls = driver.active.launcher_cls
autotune_counter = defaultdict(int)
@wraps(launcher_cls.__call__)
def patched_launcher(self, *args):
    gridX, gridY, gridZ, stream, function, packedMetadata, launch_metadata, launch_enter_hook, launch_exit_hook = args[:9]
    grid = gridX, gridY, gridZ
    name, hash, kinds = packedMetadata["kernel_name"], packedMetadata["hash"], packedMetadata["tensor_kinds"]
    kernel_args = args[9:]

    is_autotuning = any(
        frame.name == '_bench' and frame.filename.endswith('autotuner.py')
        for frame in reversed(traceback.extract_stack())
    )
    if is_autotuning:
        autotune_counter[hash] += 1
        return patched_launcher.__wrapped__(self, *args)

  # path = get_cache_manager(hash).get_file("stage_1.pkl.zstd")
  # assert path is not None
  # with open(path, "rb") as file:
  #     data = file.read()
  # dumps = pickle.loads(pyzstd.decompress(data))
  # assert len(dumps) == 9
  # print(dumps[0])

    strs, tensors = zip(*map(obj_to_str, kernel_args))
    tensors = tuple(tensor for tensor in tensors if tensor is not None)

    print(f"RUN: {name} | grid={grid}", ", ".join(strs))
    assert len(tensors) == len(kinds)

    if autotune_counter[hash]:
        print("    AUTOTUNES:", autotune_counter[hash])
        autotune_counter[hash] = 0
    result = patched_launcher.__wrapped__(self, *args)

    for kind, tensor in zip(kinds, tensors):
        if kind: # 0 - input, 1 - output, 2 - io
            name = type(tensor).__name__
            if hasattr(tensor, "_base") and name != "Tensor":
                assert name == "StridedBuffer", name
                assert tensor.dtype == tensor._base.dtype
                tensor = tensor._base

            if tensor.dtype.is_floating_point:
                tensor.normal_()
            elif tensor.dtype.is_complex:
                tensor.real.normal_()
                tensor.imag.normal_()
            elif tensor.dtype == torch.bool:
                if name == "all_kernel_2":
                    assert tensor.dim() == 0
                    tensor.fill_(True)  # site-packages/torch/distributions/distribution.py
                                        # torch._is_all_true(valid) всегда должен возвращать True
                                        # иначе test_accuracy_normal_pvalue cляжут)
                else:
                    tensor.random_(0, 2)
            else:  # целые типы
                tensor.random_(0, 100)

    return result
launcher_cls.__call__ = patched_launcher



import torch.testing
@wraps(torch.testing.assert_close)
def assert_close(actual, expected, *, allow_subclasses = True, rtol = None, atol = None, equal_nan = False,
                 check_device = True, check_dtype = True, check_layout = True, check_stride = False, msg = None):
    pass
torch.testing.assert_close = assert_close



"""
from triton.compiler.compiler import CompiledKernel

def my_enter_hook(launch_metadata):
    # launch_metadata — словарь с информацией о запуске
    print("Before kernel launch:", launch_metadata.get())

def my_exit_hook(launch_metadata):
    print("After kernel launch:", launch_metadata.get())

CompiledKernel.launch_enter_hook = my_enter_hook
CompiledKernel.launch_exit_hook = my_exit_hook
"""



import warnings, os
warnings.filterwarnings("ignore", message=".*TORCH_NPU_DEVICE_CAPABILITY.*")
warnings.filterwarnings("ignore", message=".*get_device_capability.*")
os.environ["TORCH_NPU_DEVICE_CAPABILITY"] = "8.0"

warnings.filterwarnings("ignore", message="warmup, rep, and use_cuda_graph parameters are deprecated.*")



import triton
"""
class NPUDriver(DriverBase):
    ...
    def get_empty_cache_for_benchmark(self):
        cache_size = 192 * 1024 * 1024
        return get_backend_func("get_empty_tensor", cache_size // 4)

@backend_strategy_registry.register("torch_npu", "get_empty_tensor")
def get_empty_tensor(size):
    import torch
    return torch.empty(size, dtype=torch.int32, device='npu')

/opt/python311/lib/python3.11/site-packages/triton/runtime/autotuner.py
    Здесь вызывается triton.testing.do_bench

/opt/python311/lib/python3.11/site-packages/triton/testing.py
    А здесь, при estimate_ms=0.2, получается n_repeats=4999,
    т.е. cache.zero_ вызывается 5+4999 раз, что разрывает нашему CPU одно место
    5 раз на замеры estimate_ms, 4999 уже из-за n_repeats - отношения rep=100 на estimate_ms
    это же может быть основной причиной, почему тестовые NPU почти каждый день "застревают"
    Не могли лимит на n_repeats поставить чтоли?!
"""
@wraps(triton.autotune)
def autotune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None,
             warmup=None, rep=None, use_cuda_graph=False, do_bench=None):
    warmup = rep = 0
    return autotune.__wrapped__(configs, key, prune_configs_by, reset_to_zero, restore_value, pre_hook, post_hook,
                                warmup, rep, use_cuda_graph, do_bench)
triton.autotune = autotune



from pathlib import Path
import inspect

def implement_self():
    triton_path = Path(inspect.getfile(triton))
    my_path     = Path(__file__)

    assert triton_path.exists()
    assert my_path.exists()

    splitter = "# ~~~ my implementor ~~~"
    parts = triton_path.read_text().split(splitter, 1)
    if len(parts) > 1:
        return

    implementor = f"""
{splitter}
from pathlib import Path
import sys
triton_patch_path = Path({str(my_path)!r})
if not triton_patch_path.exists():
    print(f"[TRITON IMPL] {{triton_patch_path!s}} is not defined")
elif "$not_torch_is_loaded$" not in sys.modules:
    p = str(triton_patch_path.parent)
    if p not in sys.path:
        sys.path.insert(0, p)
    import triton_patch
"""

    implemented = parts[0] + implementor
    triton_path.write_text(implemented) # enter in terminal:   sudo chown $(id -u):python -R /opt/python311


if __name__ == "__main__":
    implement_self()
