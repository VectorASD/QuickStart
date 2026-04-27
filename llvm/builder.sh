# apt install mold
set -e

get_current_hash() {
    local ref=$(git -C $1 rev-parse --symbolic-full-name HEAD)
    cat "$1/.git/$ref"
}
get_last_hash() {
    local LAST_FILE="$1/.git/info/lastHEAD"
    if [ -f "$LAST_FILE" ]; then cat "$LAST_FILE"; fi
}
set_last_hash() {
    local LAST_FILE="$1/.git/info/lastHEAD"
    echo $2 > $LAST_FILE
}

get_rebuild_flag() {
    local last_hash=$(get_last_hash $1)
    local current_hash=$(get_current_hash $1)
  # if [ "$last_hash" != "$current_hash" ]; then
    if [ "$last_hash" == "" ]; then
        echo "OLD HASH: $last_hash" >&2
        echo "NEW HASH: $current_hash" >&2
        echo "Initiate reassembly!" >&2
        set_last_hash $1 $current_hash
        echo "--rebuild"
    fi
    # else not rebuild
}

build() {
  # $1 - repository path
  # $2 - bitcode compiler (ccec) path

    cd "$1" || exit 1
    mkdir -p build && cd build
    # This step should only be executed on dev server
    local BUILD_TYPE
    if [ $(basename $1) == "hivmc" ]
        then BUILD_TYPE="Release"
        else BUILD_TYPE="Release"
    fi
    mold -run "$1/build-tools/build.sh" \
        --c-compiler clang --cxx-compiler clang++ \
        --build-type $BUILD_TYPE \
        --build ./ \
        --jobs $(nproc) \
        --fast-build \
        --enable-assertion \
        --disable-werror --disable-mlir-werror --disable-bishengir-werror \
        --build-triton \
        $(get_rebuild_flag $1) \
        --build-bishengir-template --bisheng-compiler $2 \
        --add-cmake-options "-DLLVM_PARALLEL_LINK_JOBS=1"
}

find_tail() {
    local LINE=$(grep -n "$2" "$1" | tail -n1 | cut -d: -f1)
    tail -n +"$LINE" $1
}

test() {
    local LAST_PWD=$(pwd)
    cd $1/build
    echo "cmake --build . --target 'check-bishengir' > $2"
    set +e
    cmake --build . --target 'check-bishengir' > $2
    local STATUS=$?

    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    find_tail $2 "Testing Time:"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    set -e
    cd $LAST_PWD
    echo "status: $STATUS"
    return $STATUS
}



CCEC_COMPILER="$BISHENG_INSTALL_PATH"

main() {
    REPO=$1 && shift
    git config --global --add safe.directory $REPO

    # eval "set -- $DOCKER_ARGV"   It was when it was right in .profile-1, now everything is easier :)
    while getopts "rbt" opt; do
        case "$opt" in
            r) FLAG_REBUILD="" ;;
            b) FLAG_BUILD="" ;;
            t) FLAG_TEST="" ;;
        esac
    done

    if [ -v FLAG_REBUILD ]; then
        set_last_hash $REPO ""
    fi
    if [ -v FLAG_BUILD ]; then
        build $REPO $CCEC_COMPILER
        if [ $? -eq 0 ]; then
            echo "builded is successfully"
        fi
    fi
    if [ -v FLAG_TEST ]; then
        LOG_PATH="$(dirname $REPO)/test_$(basename $REPO).log"
        test "$REPO" "$LOG_PATH"
    fi

    # cd $BISHENG/AscendNPU-IR/build; clear; cmake --build . --target 'check-bishengir' -v
    # clear; $BISHENG/AscendNPU-IR/build/bin/llvm-lit $BISHENG/AscendNPU-IR/bishengir/test/Dialect/Vector/canonicalize-2.mlir -v
    # bishengir-opt -allow-unregistered-dialect -cse -canonicalize -split-input-file
}



ensure_include() {
    local file="$1"
    local header="$2"
    local inc_line="#include <${header}>"

    # Проверка файла
    [ -f "$file" ] || { echo "Нет файла: $file"; return 1; }

    # Уже есть?
    if grep -qxF "$inc_line" "$file"; then
      # echo "OK: $inc_line уже есть в $file"
        return 0
    fi

    # Найти последний include
    local last_inc
    last_inc=$(grep -n '^#include' "$file" | tail -1 | cut -d: -f1)

    if [ -z "$last_inc" ]; then
        echo "Нет include в файле, некуда вставлять: $file"
        return 1
    fi

    # Вставить после последнего include
    sed -i "$((last_inc+1))i $inc_line" "$file"

  # echo "Добавлено: $inc_line → $file"
}

BISHENG="$HOME/bisheng"
PATH="/opt/llvm/bin:$PATH"  # 20.0.0, собран под тритон (и AscendNPU-IR без приписки '-Dev'), но хорошо работает с AscendNPU-IR-Dev

INNER_BISHENG_INSTALL_PATH=$(python -c 'from triton.backends.ascend import utils; import os; print(os.path.join(os.path.dirname(os.path.abspath(utils.__file__)), "bishengir", "bin"))')
#   А никто и не знал про этот трюк, что самое надёжное НЕ ПЕРЕЗАПИСЫВАЕМОЕ никак через env место имеено здесь :) Смотрите _get_npucompiler_path() в этом utils
#   Ранее, по этой инструкции, уже выданы права на эту директории + установлен triton

build_hivmc() {
    local REPO="$BISHENG/hivmc"
    local BIN="$INNER_BISHENG_INSTALL_PATH"  # "$BISHENG/bin"
    mkdir -p "$BIN"

    if [ -v FLAG_BUILD ]; then
        local FLAGS=""
        [ -v FLAG_REBUILD ] && ARGS="${ARGS}r"
        main $REPO "-${ARGS}b"

        echo "installing..."
        mkdir -p $BIN
        strip $REPO/build/bin/hivmc-a5 -o $BIN/hivmc-a5
        echo "installed to: $BIN"
    fi

    if [ -v FLAG_TEST ]; then
        # test after installing!
        $BISHENG/build-1 $REPO -t
    fi
}

build_bishengir_compile() {
    local REPO="$BISHENG/AscendNPU-IR-Dev"
    local BIN="$INNER_BISHENG_INSTALL_PATH"  # "$BISHENG/bin"
    mkdir -p "$BIN"

    ensure_include "$REPO/third-party/llvm-project/mlir/include/mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h" cstdint
    ensure_include "$REPO/third-party/llvm-project/mlir/include/mlir/Target/SPIRV/Deserialization.h" cstdint

    : << COMMENT
/home/vectorasd/bisheng/AscendNPU-IR-Dev/third-party/llvm-project/mlir/include/mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h:31:11: error: unknown type name 'int64_t'; did you mean '__int64_t'
это последствия использования Ubuntu-22.04 вместо Ubuntu-20.04, где убрано неявное использование заголовка на системный int64_t
#include "mlir/Support/LLVM.h" // уже было
#include <cstdint>             // добавлено

/home/vectorasd/bisheng/AscendNPU-IR-Dev/third-party/llvm-project/mlir/include/mlir/Target/SPIRV/Deserialization.h:29:51: error: use of undeclared identifier 'uint32_t'
#include "mlir/IR/OwningOpRef.h" // уже было
#include "mlir/Support/LLVM.h"   // уже было
#include <cstdint>               // добавлено

И ЭТО ВСЯ ПРОБЛЕМА Ubuntu-22.04?!
COMMENT

    if [ -v FLAG_BUILD ]; then
        local FLAGS=""
        [ -v FLAG_REBUILD ] && ARGS="${ARGS}r"
        main $REPO "-${ARGS}b"

        echo "installing..."
        mkdir -p $BIN
        strip $REPO/build/bin/bishengir-compile -o $BIN/bishengir-compile
        echo "installed to: $BIN"
    fi

    if [ -v FLAG_TEST ]; then
        # test after installing!
        $BISHENG/build-1 $REPO -t
    fi
}



eval "set -- $@"
while getopts "rbth" opt; do
    case "$opt" in
        r) FLAG_REBUILD="" ;;
        b) FLAG_BUILD="" ;;
        t) FLAG_TEST="" ;;
        h) FLAG_HIVMC="" ;;
    esac
done

# git config --global --add safe.directory $HOST_PWD
REPO_NAME=$(basename "$(git -C $HOST_PWD rev-parse --show-toplevel 2>/dev/null || true)")
if [ "$REPO_NAME" == "hivmc" ]; then FLAG_HIVMC=""; fi

if [[ -v FLAG_BUILD || -v FLAG_TEST ]]; then
    if [ -v FLAG_HIVMC ]
        then build_hivmc
        else build_bishengir_compile
    fi
    exit
fi
