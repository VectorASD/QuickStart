#include "common.h"
#include <iostream>

void __not_acl_op_compiler_placeholder() {}


#ifdef __cplusplus
extern "C" {
#endif


// cann-ge-compiler/ge-compiler/include/acl/acl_op_compiler.h

typedef enum {
    ACL_OP_COMPILE_DEFAULT = 0
} aclopCompileType;

typedef enum {
    ACL_ENGINE_SYS = 0
} aclopEngineType;

ACL_FUNC_VISIBILITY aclError aclopCompileAndExecute(const char *opType,
    int numInputs, const aclTensorDesc *const inputDesc[], const aclDataBuffer *const inputs[],
    int numOutputs, const aclTensorDesc *const outputDesc[], aclDataBuffer *const outputs[],
    const aclopAttr *attr, aclopEngineType engineType, aclopCompileType compileFlag,
    const char *opPath, aclrtStream stream) {
    std::cout << "[aclopCompileAndExecute] ";
    log_ptr("opType", opType);
    std::cout << " numInputs=" << numInputs
              << " numOutputs=" << numOutputs << " ";
    log_ptr("stream", stream);
    std::cout << std::endl;

    // not‑NPU: Torch-NPU не использует ACL Op Compiler,
    // не читает результаты, не проверяет ошибки.
    // Полный no-op.

    return ACL_SUCCESS;
}

typedef enum {
    ACL_PRECISION_MODE,
    ACL_AICORE_NUM,
    ACL_AUTO_TUNE_MODE, // The auto_tune_mode has been discarded
    ACL_OP_SELECT_IMPL_MODE,
    ACL_OPTYPELIST_FOR_IMPLMODE,
    ACL_OP_DEBUG_LEVEL,
    ACL_DEBUG_DIR,
    ACL_OP_COMPILER_CACHE_MODE,
    ACL_OP_COMPILER_CACHE_DIR,
    ACL_OP_PERFORMANCE_MODE,
    ACL_OP_JIT_COMPILE,
    ACL_OP_DETERMINISTIC,
    ACL_CUSTOMIZE_DTYPES,
    ACL_OP_PRECISION_MODE,
    ACL_ALLOW_HF32,
    ACL_PRECISION_MODE_V2,
    ACL_OP_DEBUG_OPTION
} aclCompileOpt;

ACL_FUNC_VISIBILITY aclError aclSetCompileopt(aclCompileOpt opt, const char *value) {
    std::cout << "[aclSetCompileopt] opt=" << opt << " ";
    log_ptr("value", value);
    if (value) {
        std::cout << " value_str=\"" << value << "\"";
    }
    std::cout << std::endl;

    // not‑NPU: компилятора нет, все compile options игнорируются
    return ACL_SUCCESS;
}


#ifdef __cplusplus
}
#endif
