#!/bin/bash
# Данный сборщик библиотек призван полностью заменить нестабильные not_torch.py и not_aten.py.
# Низкоуровневый API хотя бы имеет конечный объём работы, в отличие от экспоненциально растущих
#   Python‑патчей, пытающихся одновременно угодить всем слоям PyTorch.

set -e


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

SRC="$SCRIPT_DIR/src"
LIB="$SCRIPT_DIR/lib"

mkdir -p "$LIB"

# ~/QuickStart/llvm/my_acl/build_not_npu.sh
if ! grep -qxF "export LD_LIBRARY_PATH=\"$LIB:\$LD_LIBRARY_PATH\"" ~/.bashrc; then
    echo "export LD_LIBRARY_PATH=\"$LIB:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
    echo "alias build_not_npu=\"$SCRIPT_DIR/build_not_npu.sh\"" >> ~/.bashrc
    echo "reopen terminal"
    exit 1
fi
# build_not_npu && python -c "import torch_npu"   # stage 1, чтобы просто завелись библиотеки .so внутри torch_npu
# clear && build_not_npu && python -c "import torch; print(torch.randn(2, 3, device='npu'))"

# grep "aclrtSetStreamOverflowSwitch" ~/tmp/pytorch/ -rn

# git -C ~/tmp/pytorch/ submodule update --init --progress third_party/op-plugin
# grep "cmd.Name" ~/tmp/pytorch/ -rn


if [ ! -f "$LIB/libhccl.so" ]; then
    gcc -shared -fPIC "$SRC/not_hccl.c" -o "$LIB/libhccl.so"
    echo "Created lib/libhccl.so"
fi

g++ -shared -fPIC "$SRC/not_acl.cpp" -o "$LIB/libascendcl.so"
echo "Created lib/libascendcl.so"

LINK_IT=(
    -L"$LIB"
    -Wl,--no-as-needed
    -lascendcl
    -Wl,--as-needed
    -Wl,-rpath='$ORIGIN'
)
g++ -shared -fPIC "$SRC/not_acl_op_compiler.cpp" "${LINK_IT[@]}" -o "$LIB/libacl_op_compiler.so"
echo "Created lib/libacl_op_compiler.so"

if [ ! -f "$LIB/libge_runner.so" ]; then
    gcc -shared -fPIC "$SRC/not_ge_runner.c" -o "$LIB/libge_runner.so"
    echo "Created lib/libge_runner.so"
fi

if [ ! -f "$LIB/libgraph.so" ]; then
    gcc -shared -fPIC "$SRC/not_graph.c" -o "$LIB/libgraph.so"
    echo "Created lib/libgraph.so"
fi

if [ ! -f "$LIB/libacl_tdt_channel.so" ]; then
    gcc -shared -fPIC "$SRC/not_acl_tdt_channel.c" -o "$LIB/libacl_tdt_channel.so"
    echo "Created lib/libacl_tdt_channel.so"
fi
