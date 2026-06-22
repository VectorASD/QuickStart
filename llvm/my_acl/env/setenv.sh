ENV_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
SCRIPT_DIR="$(dirname "$ENV_DIR")"

LIB="$ENV_DIR/lib64"

alias build_not_npu="$SCRIPT_DIR/build_not_npu.sh"
export LD_LIBRARY_PATH="$LIB:$LD_LIBRARY_PATH"

export TRITON_NPU_COMPILER_PATH="$ENV_DIR"
export ASCEND_HOME_PATH="$ENV_DIR"
export ASCEND_OPP_PATH="$ENV_DIR"

mkdir -p "$ENV_DIR/runtime"
mkdir -p "$ENV_DIR/compiler"

cat >> "$ENV_DIR/ascend_toolkit_install.info" << EOF
version=9.1.0
EOF

export NOT_NPU_QUIET=1
python "$ENV_DIR/triton_patch.py"
