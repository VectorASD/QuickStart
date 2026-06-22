from triton.backends import backends
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

    def ttir(src, metadata):
        print("OKKKKKKKKKKKK")
        # print(src)
        return src # real_ttir(src, metadata)

    def ttadapter(src, metadata):
        return str(src) # real_ttadapter(src, metadata)

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
        metadata["tensor_kinds"] = [1] # [int(kind) for _, kind in re.findall(TENSOR_KIND_REGEX, linalg)]
        metadata["required_ub_bits"] = 0
        bitcodes = re.findall(BITCODES_REGEX, linalg)
        metadata["bitcodes"] = [val for group in bitcodes for val in group if val]
        return linalg # real_npubin(linalg, metadata)

    stages.clear()
    stages["ttir"]      = ttir
    stages["ttadapter"] = ttadapter
    stages["npubin"]    = npubin
AscendBackend.add_stages = add_stages



from pathlib import Path
import inspect
import triton

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
