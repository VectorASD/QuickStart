#!/bin/bash
# Данный сборщик библиотек призван полностью заменить нестабильные not_torch.py и not_aten.py.
# Низкоуровневый API хотя бы имеет конечный объём работы, в отличие от экспоненциально растущих
#   Python‑патчей, пытающихся одновременно угодить всем слоям PyTorch.

set -e


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

SRC="$SCRIPT_DIR/src"
OBJ="$SCRIPT_DIR/obj"
LIB="$SCRIPT_DIR/lib"

mkdir -p "$OBJ" "$LIB"


: << 'COMMENT'
TORCH_INCLUDE=$(python -c "import torch; from torch.utils.cpp_extension import include_paths; print(include_paths()[0])")
TORCH_LIB=$(python -c "import torch; from torch.utils.cpp_extension import library_paths; print(library_paths()[0])")
echo "TORCH_INCLUDE=$TORCH_INCLUDE"
echo "TORCH_LIB=$TORCH_LIB"
COMMENT
TORCH_INCLUDE=/opt/python311/lib/python3.11/site-packages/torch/include
TORCH_LIB=/opt/python311/lib/python3.11/site-packages/torch/lib
TORCH_FLAGS=(
    -I"$TORCH_INCLUDE"
    -I"$TORCH_INCLUDE/torch/csrc/api/include"
    -L"$TORCH_LIB"
    -ltorch -ltorch_cpu -lc10
)

# ~/QuickStart/llvm/my_acl/build_not_npu.sh
if ! grep -qxF "export LD_LIBRARY_PATH=\"$LIB:\$LD_LIBRARY_PATH\"" ~/.bashrc; then
    echo "export LD_LIBRARY_PATH=\"$LIB:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
    echo "alias build_not_npu=\"$SCRIPT_DIR/build_not_npu.sh\"" >> ~/.bashrc
    echo "reopen terminal"
    exit 1
fi
: << 'COMMENT'
LEVEL 1:
    build_not_npu && python -c "import torch_npu"   # stage 1, чтобы просто завелись библиотеки .so внутри torch_npu
LEVEL 2:
    clear && build_not_npu && python -c "import torch; print(torch.randn(2, 3, device='npu'))"
LEVEL 3:
    clear && build_not_npu && python -c "import torch; print(torch.randn(32, 32, device='npu', dtype=torch.float16))"
LEVEL 4:
    sudo apt install valgrind
    Добавлен флаг -g к g++
    clear && valgrind --leak-check=full pytest test_tensor_constructor_ops.py -sv
        Утечек памяти не обнаружено. Максимум пару МБ за всё время, и то, только в момент загрузки библиотек
    clear && valgrind --tool=massif --massif-out-file=massif.out pytest test_tensor_constructor_ops.py -sv
        Сильная нагрузка на стек, в основном в момент torch.eye. Вероятнее всего, тензоры просто не успевают освободится
        К примеру, torch.eye(8192, dtype=torch.double) весит 0.5 GB
    for i in range(100): torch.eye(8192, device="npu", dtype=torch.double)
        Не очень похоже на то, чтобы у torch_npu и моего ACL были проблемы с освобождением памяти
    Виноват pytest?!
        Нет.
    Решение найдено:
        ctypes.CDLL("libc.so.6").malloc_trim(0)

grep "aclrtSetStreamOverflowSwitch" ~/tmp/pytorch/ -rn

git -C ~/tmp/pytorch/ submodule update --init --progress third_party/op-plugin
grep "cmd.Name" ~/tmp/pytorch/ -rn
COMMENT


export CC="ccache gcc"
export CXX="ccache g++"
export MOLD_FLAGS="-fuse-ld=mold"

# 1. Компиляция объектных файлов (ccache кеширует .o)
$CXX    -fPIC -c "$SRC/not_acl.cpp"               -I"$SRC" -o "$OBJ/not_acl.o"                               && echo "not_acl.o             DONE"
$CXX    -fPIC -c "$SRC/not_acl_op_compiler.cpp"   -I"$SRC" ${TORCH_FLAGS[@]} -o "$OBJ/not_acl_op_compiler.o" && echo "not_acl_op_compiler.o DONE"
$CXX    -fPIC -c "$SRC/not_opapi.cpp"             -I"$SRC" ${TORCH_FLAGS[@]} -o "$OBJ/not_opapi.o"           && echo "not_opapi.o           DONE"

$CC     -fPIC -c "$SRC/not_hccl.c"                -o "$OBJ/not_hccl.o"
$CC     -fPIC -c "$SRC/not_ge_runner.c"           -o "$OBJ/not_ge_runner.o"
$CC     -fPIC -c "$SRC/not_graph.c"               -o "$OBJ/not_graph.o"
$CC     -fPIC -c "$SRC/not_acl_tdt_channel.c"     -o "$OBJ/not_acl_tdt_channel.o"

# 2. Линковка разделяемых библиотек (mold ускоряет)
echo "Linking shared libraries..."
$CC  $MOLD_FLAGS -shared "$OBJ/not_hccl.o"               -o "$LIB/libhccl.so"
$CXX $MOLD_FLAGS -shared "$OBJ/not_acl.o"                -o "$LIB/libascendcl.so"
$CXX $MOLD_FLAGS -shared "$OBJ/not_opapi.o"              ${TORCH_FLAGS[@]} -o "$LIB/libopapi.so"
$CXX $MOLD_FLAGS -shared "$OBJ/not_acl_op_compiler.o"   \
    -L"$LIB" -Wl,--no-as-needed -lascendcl -Wl,--as-needed \
    -Wl,-rpath='$ORIGIN' ${TORCH_FLAGS[@]} -o "$LIB/libacl_op_compiler.so"
$CC  $MOLD_FLAGS -shared "$OBJ/not_ge_runner.o"          -o "$LIB/libge_runner.so"
$CC  $MOLD_FLAGS -shared "$OBJ/not_graph.o"              -o "$LIB/libgraph.so"
$CC  $MOLD_FLAGS -shared "$OBJ/not_acl_tdt_channel.o"    -o "$LIB/libacl_tdt_channel.so"
