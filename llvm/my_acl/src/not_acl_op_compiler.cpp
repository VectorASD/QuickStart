#include "common.h"
#include "not_acl.cpp"  // aclGetTensorDescType
#include "helpers.cpp"  // calc_num_elements, formatTensorList
#include <cstring>      // memcpy, memset, size_t, strstr

void __not_acl_op_compiler_placeholder() {}


#ifdef __cplusplus
extern "C" {
#endif


// cann-ge-compiler/ge-compiler/include/acl/acl_op_compiler.h

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

static const char* aclCompileOptToString(aclCompileOpt opt) {
    switch (opt) {
        case ACL_PRECISION_MODE:            return "PRECISION_MODE";
        case ACL_AICORE_NUM:                return "AICORE_NUM";
        case ACL_AUTO_TUNE_MODE:            return "AUTO_TUNE_MODE";
        case ACL_OP_SELECT_IMPL_MODE:       return "OP_SELECT_IMPL_MODE";
        case ACL_OPTYPELIST_FOR_IMPLMODE:   return "OPTYPELIST_FOR_IMPLMODE";
        case ACL_OP_DEBUG_LEVEL:            return "OP_DEBUG_LEVEL";
        case ACL_DEBUG_DIR:                 return "DEBUG_DIR";
        case ACL_OP_COMPILER_CACHE_MODE:    return "OP_COMPILER_CACHE_MODE";
        case ACL_OP_COMPILER_CACHE_DIR:     return "OP_COMPILER_CACHE_DIR";
        case ACL_OP_PERFORMANCE_MODE:       return "OP_PERFORMANCE_MODE";
        case ACL_OP_JIT_COMPILE:            return "OP_JIT_COMPILE";
        case ACL_OP_DETERMINISTIC:          return "OP_DETERMINISTIC";
        case ACL_CUSTOMIZE_DTYPES:           return "CUSTOMIZE_DTYPES";
        case ACL_OP_PRECISION_MODE:         return "OP_PRECISION_MODE";
        case ACL_ALLOW_HF32:                return "ALLOW_HF32";
        case ACL_PRECISION_MODE_V2:         return "PRECISION_MODE_V2";
        case ACL_OP_DEBUG_OPTION:           return "OP_DEBUG_OPTION";
        default:                            return "UNKNOWN";
    }
}

ACL_FUNC_VISIBILITY aclError aclSetCompileopt(aclCompileOpt opt, const char *value) {
    std::ostringstream log;
    log << "[aclSetCompileopt] opt=" << aclCompileOptToString(opt)
        << " (" << static_cast<int>(opt) << ")"
        << " value=" << (value ? value : "(null)");
    log_output(log);

    // not‑NPU: компилятора нет, все compile options игнорируются
    return ACL_SUCCESS;
}


// aclopCompileAndExecute функционал

typedef enum {
    ACL_OP_COMPILE_DEFAULT = 0
} aclopCompileType;

typedef enum {
    ACL_ENGINE_SYS = 0
} aclopEngineType;


REGISTER_OP(StatelessRandomNormalV2, {
    if (numInputs != 4 || numOutputs != 1)
        return H_UNASSERTED;
    // Проверка типов входов
    if (!inputDesc[1] || aclGetTensorDescType(inputDesc[1], false) != ACL_UINT64)
        return H_UNASSERTED;
    if (!inputDesc[2] || aclGetTensorDescType(inputDesc[2], false) != ACL_UINT64)
        return H_UNASSERTED;
    if (!inputDesc[3] || aclGetTensorDescType(inputDesc[3], false) != ACL_INT32)
        return H_UNASSERTED;

    // Извлечение seed
    uint64_t seed_value = 0;
    bool has_seed = false;
    if (inputs[1] && inputDesc[1] && inputs[1]->data)
        if (aclGetTensorDescType(inputDesc[1], false) == ACL_UINT64) {
            size_t seed_count = calc_num_elements(inputDesc[1], inputs[1]->size);
            if (seed_count >= 1) {
                const uint64_t* seed_ptr = static_cast<const uint64_t*>(inputs[1]->data);
                seed_value = seed_ptr[0];
                has_seed = true;
            }
        }

    std::mt19937 local_rng;
    if (has_seed)
        local_rng.seed(static_cast<std::mt19937::result_type>(seed_value));
    else {
        std::random_device rd;
        local_rng.seed(rd());
    }

    if (outputs[0] && outputDesc[0] && outputs[0]->data) {
        aclDataType dt = aclGetTensorDescType(outputDesc[0], false);
        switch (dt) {
        DISPATCH_RANDOM(ACL_FLOAT)
        DISPATCH_RANDOM(ACL_DOUBLE)
        DISPATCH_RANDOM(ACL_FLOAT16)
        DISPATCH_RANDOM(ACL_BF16)
        DISPATCH_RANDOM(ACL_INT8)
        DISPATCH_RANDOM(ACL_UINT8)
        DISPATCH_RANDOM(ACL_INT16)
        DISPATCH_RANDOM(ACL_UINT16)
        DISPATCH_RANDOM(ACL_INT32)
        DISPATCH_RANDOM(ACL_UINT32)
        DISPATCH_RANDOM(ACL_INT64)
        DISPATCH_RANDOM(ACL_UINT64)
        DISPATCH_RANDOM(ACL_BOOL)
        default:
            return H_UNIMPLEMENTED; // тип не поддерживается
        }
    } else
        return H_UNASSERTED; // Нет данных в выходе — тоже ошибка
    return H_OK;
});

REGISTER_OP(StatelessRandomUniformV2, {
    if (numInputs != 4 || numOutputs != 1)
        return H_UNASSERTED;
    if (!inputDesc[1] || aclGetTensorDescType(inputDesc[1], false) != ACL_UINT64)
        return H_UNASSERTED;
    if (!inputDesc[2] || aclGetTensorDescType(inputDesc[2], false) != ACL_UINT64)
        return H_UNASSERTED;
    if (!inputDesc[3] || aclGetTensorDescType(inputDesc[3], false) != ACL_INT32)
        return H_UNASSERTED;

    uint64_t seed_value = 0;
    if (inputs[1] && inputs[1]->data) {
        const auto* seed_ptr = static_cast<const uint64_t*>(inputs[1]->data);
        seed_value = seed_ptr[0];
    }

    std::mt19937 local_rng;
    local_rng.seed(static_cast<std::mt19937::result_type>(seed_value));
    if (outputs[0] && outputDesc[0] && outputs[0]->data) {
        aclDataType dt = aclGetTensorDescType(outputDesc[0], false);
        switch (dt) {
            DISPATCH_RANDOM_UNIFORM(ACL_FLOAT)
            DISPATCH_RANDOM_UNIFORM(ACL_DOUBLE)
            DISPATCH_RANDOM_UNIFORM(ACL_FLOAT16)
            DISPATCH_RANDOM_UNIFORM(ACL_BF16)
            DISPATCH_RANDOM_UNIFORM(ACL_INT8)
            DISPATCH_RANDOM_UNIFORM(ACL_UINT8)
            DISPATCH_RANDOM_UNIFORM(ACL_INT16)
            DISPATCH_RANDOM_UNIFORM(ACL_UINT16)
            DISPATCH_RANDOM_UNIFORM(ACL_INT32)
            DISPATCH_RANDOM_UNIFORM(ACL_UINT32)
            DISPATCH_RANDOM_UNIFORM(ACL_INT64)
            DISPATCH_RANDOM_UNIFORM(ACL_UINT64)
            DISPATCH_RANDOM_UNIFORM(ACL_BOOL)
            default:
                return H_UNIMPLEMENTED;
        }
    } else
        return H_UNIMPLEMENTED;
    return H_OK;
});


REGISTER_OP(ZerosLike, {
    if (numInputs != 1 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;

    aclDataType dt = aclGetTensorDescType(outputDesc[0], false);
    switch (dt) {
        DISPATCH_ZEROS_LIKE(ACL_FLOAT)
        DISPATCH_ZEROS_LIKE(ACL_DOUBLE)
        DISPATCH_ZEROS_LIKE(ACL_FLOAT16)
        DISPATCH_ZEROS_LIKE(ACL_BF16)
        DISPATCH_ZEROS_LIKE(ACL_INT8)
        DISPATCH_ZEROS_LIKE(ACL_UINT8)
        DISPATCH_ZEROS_LIKE(ACL_INT16)
        DISPATCH_ZEROS_LIKE(ACL_UINT16)
        DISPATCH_ZEROS_LIKE(ACL_INT32)
        DISPATCH_ZEROS_LIKE(ACL_UINT32)
        DISPATCH_ZEROS_LIKE(ACL_INT64)
        DISPATCH_ZEROS_LIKE(ACL_UINT64)
        DISPATCH_ZEROS_LIKE(ACL_BOOL)
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(Fill, {
    if (numInputs != 2 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[1] || !inputDesc[1] || !inputs[1]->data)
        return H_UNASSERTED;

    aclDataType outDt = aclGetTensorDescType(outputDesc[0], false);
    switch (outDt) {
        DISPATCH_FILL(ACL_FLOAT)
        DISPATCH_FILL(ACL_DOUBLE)
        DISPATCH_FILL(ACL_FLOAT16)
        DISPATCH_FILL(ACL_BF16)
        DISPATCH_FILL(ACL_INT8)
        DISPATCH_FILL(ACL_UINT8)
        DISPATCH_FILL(ACL_INT16)
        DISPATCH_FILL(ACL_UINT16)
        DISPATCH_FILL(ACL_INT32)
        DISPATCH_FILL(ACL_UINT32)
        DISPATCH_FILL(ACL_INT64)
        DISPATCH_FILL(ACL_UINT64)
        DISPATCH_FILL(ACL_BOOL)
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(Mul, {
    if (numInputs != 2 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[1] || !inputDesc[1] || !inputs[1]->data)
        return H_UNASSERTED;

    aclDataType dt = aclGetTensorDescType(outputDesc[0], false);
    switch (dt) {
        DISPATCH_MUL(ACL_FLOAT)
        DISPATCH_MUL(ACL_DOUBLE)
        DISPATCH_MUL(ACL_FLOAT16)
        DISPATCH_MUL(ACL_BF16)
        DISPATCH_MUL(ACL_INT8)
        DISPATCH_MUL(ACL_UINT8)
        DISPATCH_MUL(ACL_INT16)
        DISPATCH_MUL(ACL_UINT16)
        DISPATCH_MUL(ACL_INT32)
        DISPATCH_MUL(ACL_UINT32)
        DISPATCH_MUL(ACL_INT64)
        DISPATCH_MUL(ACL_UINT64)
        // ACL_BOOL для Mul не поддерживается в torch_npu (для bool используется LogicalAnd),
        // поэтому если он сюда попадёт, уйдём в H_UNIMPLEMENTED
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(Add, {
    if (numInputs != 2 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[1] || !inputDesc[1] || !inputs[1]->data)
        return H_UNASSERTED;

    aclDataType dt = aclGetTensorDescType(outputDesc[0], false);
    switch (dt) {
        DISPATCH_ADD(ACL_FLOAT)
        DISPATCH_ADD(ACL_DOUBLE)
        DISPATCH_ADD(ACL_FLOAT16)
        DISPATCH_ADD(ACL_BF16)
        DISPATCH_ADD(ACL_INT8)
        DISPATCH_ADD(ACL_UINT8)
        DISPATCH_ADD(ACL_INT16)
        DISPATCH_ADD(ACL_UINT16)
        DISPATCH_ADD(ACL_INT32)
        DISPATCH_ADD(ACL_UINT32)
        DISPATCH_ADD(ACL_INT64)
        DISPATCH_ADD(ACL_UINT64)
        // ACL_BOOL для Add не поддерживается в torch_npu (для bool используется LogicalOr),
        // поэтому если он сюда попадёт, уйдём в H_UNIMPLEMENTED
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(Sub, {
    if (numInputs != 2 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[1] || !inputDesc[1] || !inputs[1]->data)
        return H_UNASSERTED;

    aclDataType dt = aclGetTensorDescType(outputDesc[0], false);
    switch (dt) {
        DISPATCH_SUB(ACL_FLOAT)
        DISPATCH_SUB(ACL_DOUBLE)
        DISPATCH_SUB(ACL_FLOAT16)
        DISPATCH_SUB(ACL_BF16)
        DISPATCH_SUB(ACL_INT8)
        DISPATCH_SUB(ACL_UINT8)
        DISPATCH_SUB(ACL_INT16)
        DISPATCH_SUB(ACL_UINT16)
        DISPATCH_SUB(ACL_INT32)
        DISPATCH_SUB(ACL_UINT32)
        DISPATCH_SUB(ACL_INT64)
        DISPATCH_SUB(ACL_UINT64)
        // ACL_BOOL не используется для Sub
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(RealDiv, {
    if (numInputs != 2 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[1] || !inputDesc[1] || !inputs[1]->data)
        return H_UNASSERTED;

    aclDataType dt = aclGetTensorDescType(outputDesc[0], false);
    switch (dt) {
        DISPATCH_DIV(ACL_FLOAT)
        DISPATCH_DIV(ACL_DOUBLE)
        DISPATCH_DIV(ACL_FLOAT16)
        DISPATCH_DIV(ACL_BF16)
        // целые типы — с защитой от деления на ноль (всегда даст 0)
        DISPATCH_DIV(ACL_INT8)
        DISPATCH_DIV(ACL_UINT8)
        DISPATCH_DIV(ACL_INT16)
        DISPATCH_DIV(ACL_UINT16)
        DISPATCH_DIV(ACL_INT32)
        DISPATCH_DIV(ACL_UINT32)
        DISPATCH_DIV(ACL_INT64)
        DISPATCH_DIV(ACL_UINT64)
        // ACL_BOOL для RealDiv не используется
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(IsFinite, {
    if (numInputs != 1 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;

    aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
    switch (inDt) {
        DISPATCH_ISFINITE(ACL_FLOAT)
        DISPATCH_ISFINITE(ACL_DOUBLE)
        DISPATCH_ISFINITE(ACL_FLOAT16)
        DISPATCH_ISFINITE(ACL_BF16)
        DISPATCH_ISFINITE(ACL_INT8)
        DISPATCH_ISFINITE(ACL_UINT8)
        DISPATCH_ISFINITE(ACL_INT16)
        DISPATCH_ISFINITE(ACL_UINT16)
        DISPATCH_ISFINITE(ACL_INT32)
        DISPATCH_ISFINITE(ACL_UINT32)
        DISPATCH_ISFINITE(ACL_INT64)
        DISPATCH_ISFINITE(ACL_UINT64)
        DISPATCH_ISFINITE(ACL_BOOL)
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(Equal, {
    if (numInputs != 2 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[1] || !inputDesc[1] || !inputs[1]->data)
        return H_UNASSERTED;

    aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
    switch (inDt) {
        DISPATCH_COMPARE(ACL_FLOAT,    std::equal_to<float>{})
        DISPATCH_COMPARE(ACL_DOUBLE,   std::equal_to<double>{})
        DISPATCH_COMPARE(ACL_INT8,     std::equal_to<int8_t>{})
        DISPATCH_COMPARE(ACL_UINT8,    std::equal_to<uint8_t>{})
        DISPATCH_COMPARE(ACL_INT16,    std::equal_to<int16_t>{})
        DISPATCH_COMPARE(ACL_UINT16,   std::equal_to<uint16_t>{})
        DISPATCH_COMPARE(ACL_INT32,    std::equal_to<int32_t>{})
        DISPATCH_COMPARE(ACL_UINT32,   std::equal_to<uint32_t>{})
        DISPATCH_COMPARE(ACL_INT64,    std::equal_to<int64_t>{})
        DISPATCH_COMPARE(ACL_UINT64,   std::equal_to<uint64_t>{})
        DISPATCH_COMPARE(ACL_FLOAT16,  [](uint16_t a, uint16_t b) { return half_to_float(a) == half_to_float(b); })
        DISPATCH_COMPARE(ACL_BF16,     [](uint16_t a, uint16_t b) { return bf16_to_float(a) == bf16_to_float(b); })
        DISPATCH_COMPARE(ACL_BOOL,     std::equal_to<bool>{})
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(NotEqual, {
    if (numInputs != 2 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[1] || !inputDesc[1] || !inputs[1]->data)
        return H_UNASSERTED;

    aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
    switch (inDt) {
        DISPATCH_COMPARE(ACL_FLOAT,    std::not_equal_to<float>{})
        DISPATCH_COMPARE(ACL_DOUBLE,   std::not_equal_to<double>{})
        DISPATCH_COMPARE(ACL_INT8,     std::not_equal_to<int8_t>{})
        DISPATCH_COMPARE(ACL_UINT8,    std::not_equal_to<uint8_t>{})
        DISPATCH_COMPARE(ACL_INT16,    std::not_equal_to<int16_t>{})
        DISPATCH_COMPARE(ACL_UINT16,   std::not_equal_to<uint16_t>{})
        DISPATCH_COMPARE(ACL_INT32,    std::not_equal_to<int32_t>{})
        DISPATCH_COMPARE(ACL_UINT32,   std::not_equal_to<uint32_t>{})
        DISPATCH_COMPARE(ACL_INT64,    std::not_equal_to<int64_t>{})
        DISPATCH_COMPARE(ACL_UINT64,   std::not_equal_to<uint64_t>{})
        DISPATCH_COMPARE(ACL_FLOAT16,  [](uint16_t a, uint16_t b) { return half_to_float(a) != half_to_float(b); })
        DISPATCH_COMPARE(ACL_BF16,     [](uint16_t a, uint16_t b) { return bf16_to_float(a) != bf16_to_float(b); })
        DISPATCH_COMPARE(ACL_BOOL,     std::not_equal_to<bool>{})
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(TensorEqual, {
    if (numInputs != 2 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[1] || !inputDesc[1] || !inputs[1]->data)
        return H_UNASSERTED;

    int64_t dim;
    if (aclGetTensorDescType(outputDesc[0], false) != ACL_BOOL ||
        aclGetTensorDescNumDims(outputDesc[0], false) != 1 ||
        aclGetTensorDescDimV2(outputDesc[0], 0, &dim, false) != ACL_SUCCESS || dim != 1)
        return H_UNASSERTED;

    aclDataType dt = aclGetTensorDescType(inputDesc[0], false);
    if (dt != aclGetTensorDescType(inputDesc[1], false))
        return H_UNASSERTED;

    switch (dt) {
        DISPATCH_TENSOR_EQUAL(ACL_FLOAT)
        DISPATCH_TENSOR_EQUAL(ACL_DOUBLE)
        DISPATCH_TENSOR_EQUAL(ACL_INT8)
        DISPATCH_TENSOR_EQUAL(ACL_UINT8)
        DISPATCH_TENSOR_EQUAL(ACL_INT16)
        DISPATCH_TENSOR_EQUAL(ACL_UINT16)
        DISPATCH_TENSOR_EQUAL(ACL_INT32)
        DISPATCH_TENSOR_EQUAL(ACL_UINT32)
        DISPATCH_TENSOR_EQUAL(ACL_INT64)
        DISPATCH_TENSOR_EQUAL(ACL_UINT64)
        DISPATCH_TENSOR_EQUAL(ACL_BOOL)
        DISPATCH_TENSOR_EQUAL(ACL_FLOAT16)
        DISPATCH_TENSOR_EQUAL(ACL_BF16)
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(Greater, {
    if (numInputs != 2 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[1] || !inputDesc[1] || !inputs[1]->data)
        return H_UNASSERTED;

    aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
    switch (inDt) {
        DISPATCH_COMPARE(ACL_FLOAT,    std::greater<float>{})
        DISPATCH_COMPARE(ACL_DOUBLE,   std::greater<double>{})
        DISPATCH_COMPARE(ACL_INT8,     std::greater<int8_t>{})
        DISPATCH_COMPARE(ACL_UINT8,    std::greater<uint8_t>{})
        DISPATCH_COMPARE(ACL_INT16,    std::greater<int16_t>{})
        DISPATCH_COMPARE(ACL_UINT16,   std::greater<uint16_t>{})
        DISPATCH_COMPARE(ACL_INT32,    std::greater<int32_t>{})
        DISPATCH_COMPARE(ACL_UINT32,   std::greater<uint32_t>{})
        DISPATCH_COMPARE(ACL_INT64,    std::greater<int64_t>{})
        DISPATCH_COMPARE(ACL_UINT64,   std::greater<uint64_t>{})
        DISPATCH_COMPARE(ACL_FLOAT16,  [](uint16_t a, uint16_t b) { return half_to_float(a) > half_to_float(b); })
        DISPATCH_COMPARE(ACL_BF16,     [](uint16_t a, uint16_t b) { return bf16_to_float(a) > bf16_to_float(b); })
        DISPATCH_COMPARE(ACL_BOOL,     std::greater<bool>{})
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(GreaterEqual, {
    if (numInputs != 2 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[1] || !inputDesc[1] || !inputs[1]->data)
        return H_UNASSERTED;

    aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
    switch (inDt) {
        DISPATCH_COMPARE(ACL_FLOAT,    std::greater_equal<float>{})
        DISPATCH_COMPARE(ACL_DOUBLE,   std::greater_equal<double>{})
        DISPATCH_COMPARE(ACL_INT8,     std::greater_equal<int8_t>{})
        DISPATCH_COMPARE(ACL_UINT8,    std::greater_equal<uint8_t>{})
        DISPATCH_COMPARE(ACL_INT16,    std::greater_equal<int16_t>{})
        DISPATCH_COMPARE(ACL_UINT16,   std::greater_equal<uint16_t>{})
        DISPATCH_COMPARE(ACL_INT32,    std::greater_equal<int32_t>{})
        DISPATCH_COMPARE(ACL_UINT32,   std::greater_equal<uint32_t>{})
        DISPATCH_COMPARE(ACL_INT64,    std::greater_equal<int64_t>{})
        DISPATCH_COMPARE(ACL_UINT64,   std::greater_equal<uint64_t>{})
        DISPATCH_COMPARE(ACL_FLOAT16,  [](uint16_t a, uint16_t b) { return half_to_float(a) >= half_to_float(b); })
        DISPATCH_COMPARE(ACL_BF16,     [](uint16_t a, uint16_t b) { return bf16_to_float(a) >= bf16_to_float(b); })
        DISPATCH_COMPARE(ACL_BOOL,     std::greater_equal<bool>{})
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(Less, {
    if (numInputs != 2 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[1] || !inputDesc[1] || !inputs[1]->data)
        return H_UNASSERTED;

    aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
    switch (inDt) {
        DISPATCH_COMPARE(ACL_FLOAT,    std::less<float>{})
        DISPATCH_COMPARE(ACL_DOUBLE,   std::less<double>{})
        DISPATCH_COMPARE(ACL_INT8,     std::less<int8_t>{})
        DISPATCH_COMPARE(ACL_UINT8,    std::less<uint8_t>{})
        DISPATCH_COMPARE(ACL_INT16,    std::less<int16_t>{})
        DISPATCH_COMPARE(ACL_UINT16,   std::less<uint16_t>{})
        DISPATCH_COMPARE(ACL_INT32,    std::less<int32_t>{})
        DISPATCH_COMPARE(ACL_UINT32,   std::less<uint32_t>{})
        DISPATCH_COMPARE(ACL_INT64,    std::less<int64_t>{})
        DISPATCH_COMPARE(ACL_UINT64,   std::less<uint64_t>{})
        DISPATCH_COMPARE(ACL_FLOAT16,  [](uint16_t a, uint16_t b) { return half_to_float(a) < half_to_float(b); })
        DISPATCH_COMPARE(ACL_BF16,     [](uint16_t a, uint16_t b) { return bf16_to_float(a) < bf16_to_float(b); })
        DISPATCH_COMPARE(ACL_BOOL,     std::less<bool>{})
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(LessEqual, {
    if (numInputs != 2 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[1] || !inputDesc[1] || !inputs[1]->data)
        return H_UNASSERTED;

    aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
    switch (inDt) {
        DISPATCH_COMPARE(ACL_FLOAT,    std::less_equal<float>{})
        DISPATCH_COMPARE(ACL_DOUBLE,   std::less_equal<double>{})
        DISPATCH_COMPARE(ACL_INT8,     std::less_equal<int8_t>{})
        DISPATCH_COMPARE(ACL_UINT8,    std::less_equal<uint8_t>{})
        DISPATCH_COMPARE(ACL_INT16,    std::less_equal<int16_t>{})
        DISPATCH_COMPARE(ACL_UINT16,   std::less_equal<uint16_t>{})
        DISPATCH_COMPARE(ACL_INT32,    std::less_equal<int32_t>{})
        DISPATCH_COMPARE(ACL_UINT32,   std::less_equal<uint32_t>{})
        DISPATCH_COMPARE(ACL_INT64,    std::less_equal<int64_t>{})
        DISPATCH_COMPARE(ACL_UINT64,   std::less_equal<uint64_t>{})
        DISPATCH_COMPARE(ACL_FLOAT16,  [](uint16_t a, uint16_t b) { return half_to_float(a) <= half_to_float(b); })
        DISPATCH_COMPARE(ACL_BF16,     [](uint16_t a, uint16_t b) { return bf16_to_float(a) <= bf16_to_float(b); })
        DISPATCH_COMPARE(ACL_BOOL,     std::less_equal<bool>{})
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(Cast, {
    if (numInputs != 1 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;

    aclDataType inDt  = aclGetTensorDescType(inputDesc[0], false);
    aclDataType outDt = aclGetTensorDescType(outputDesc[0], false);
    size_t count = calc_num_elements(outputDesc[0], outputs[0]->size);
    if (count == 0) return H_OK;

    // Одинаковые типы – просто копируем память
    if (inDt == outDt) {
        size_t elemSize = aclDataTypeBytes(inDt);
        memcpy(outputs[0]->data, inputs[0]->data, count * elemSize);
        return H_OK;
    }

    // Проверка на поддерживаемые типы (без экзотики)
    auto isSupported = [](aclDataType dt) {
        switch (dt) {
            case ACL_FLOAT: case ACL_DOUBLE: case ACL_FLOAT16: case ACL_BF16:
            case ACL_INT8: case ACL_UINT8: case ACL_INT16: case ACL_UINT16:
            case ACL_INT32: case ACL_UINT32: case ACL_INT64: case ACL_UINT64:
            case ACL_BOOL:
                return true;
            default:
                return false;
        }
    };
    if (!isSupported(inDt) || !isSupported(outDt))
        return H_UNIMPLEMENTED;

    // Функция чтения элемента как double (для 64-битных) или float (для остальных)
    auto readAsDouble = [&](size_t i) -> double {
        switch (inDt) {
            // 64-битные целые и double читаем сразу как double (без потерь)
            case ACL_INT64:   return static_cast<double>(static_cast<const int64_t*>(inputs[0]->data)[i]);
            case ACL_UINT64:  return static_cast<double>(static_cast<const uint64_t*>(inputs[0]->data)[i]);
            case ACL_DOUBLE:  return static_cast<const double*>(inputs[0]->data)[i];
            // Остальные преобразуем через float – потерь точности для них нет
            default: {
                float val = 0.0f;
                switch (inDt) {
                    case ACL_FLOAT:   val = aclDataTypeTraits<ACL_FLOAT>::to_float(static_cast<const float*>(inputs[0]->data)[i]); break;
                    case ACL_FLOAT16: val = aclDataTypeTraits<ACL_FLOAT16>::to_float(static_cast<const uint16_t*>(inputs[0]->data)[i]); break;
                    case ACL_BF16:    val = aclDataTypeTraits<ACL_BF16>::to_float(static_cast<const uint16_t*>(inputs[0]->data)[i]); break;
                    case ACL_INT8:    val = aclDataTypeTraits<ACL_INT8>::to_float(static_cast<const int8_t*>(inputs[0]->data)[i]); break;
                    case ACL_UINT8:   val = aclDataTypeTraits<ACL_UINT8>::to_float(static_cast<const uint8_t*>(inputs[0]->data)[i]); break;
                    case ACL_INT16:   val = aclDataTypeTraits<ACL_INT16>::to_float(static_cast<const int16_t*>(inputs[0]->data)[i]); break;
                    case ACL_UINT16:  val = aclDataTypeTraits<ACL_UINT16>::to_float(static_cast<const uint16_t*>(inputs[0]->data)[i]); break;
                    case ACL_INT32:   val = aclDataTypeTraits<ACL_INT32>::to_float(static_cast<const int32_t*>(inputs[0]->data)[i]); break;
                    case ACL_UINT32:  val = aclDataTypeTraits<ACL_UINT32>::to_float(static_cast<const uint32_t*>(inputs[0]->data)[i]); break;
                    case ACL_BOOL:    val = aclDataTypeTraits<ACL_BOOL>::to_float(static_cast<const bool*>(inputs[0]->data)[i]); break;
                    default: break;
                }
                return static_cast<double>(val);
            }
        }
    };

    // Функция записи double в выходной тип
    auto writeFromDouble = [&](size_t i, double d) {
        switch (outDt) {
            case ACL_INT64:   static_cast<int64_t*>(outputs[0]->data)[i]   = static_cast<int64_t>(d); break;
            case ACL_UINT64:  static_cast<uint64_t*>(outputs[0]->data)[i]  = static_cast<uint64_t>(d); break;
            case ACL_DOUBLE:  static_cast<double*>(outputs[0]->data)[i]    = d; break;
            default: {
                // для всех остальных типов сужаем до float и вызываем from_float
                float val = static_cast<float>(d);
                switch (outDt) {
                    case ACL_FLOAT:   static_cast<float*>(outputs[0]->data)[i] = aclDataTypeTraits<ACL_FLOAT>::from_float(val); break;
                    case ACL_FLOAT16: static_cast<uint16_t*>(outputs[0]->data)[i] = aclDataTypeTraits<ACL_FLOAT16>::from_float(val); break;
                    case ACL_BF16:    static_cast<uint16_t*>(outputs[0]->data)[i] = aclDataTypeTraits<ACL_BF16>::from_float(val); break;
                    case ACL_INT8:    static_cast<int8_t*>(outputs[0]->data)[i]   = aclDataTypeTraits<ACL_INT8>::from_float(val); break;
                    case ACL_UINT8:   static_cast<uint8_t*>(outputs[0]->data)[i]  = aclDataTypeTraits<ACL_UINT8>::from_float(val); break;
                    case ACL_INT16:   static_cast<int16_t*>(outputs[0]->data)[i]  = aclDataTypeTraits<ACL_INT16>::from_float(val); break;
                    case ACL_UINT16:  static_cast<uint16_t*>(outputs[0]->data)[i] = aclDataTypeTraits<ACL_UINT16>::from_float(val); break;
                    case ACL_INT32:   static_cast<int32_t*>(outputs[0]->data)[i]  = aclDataTypeTraits<ACL_INT32>::from_float(val); break;
                    case ACL_UINT32:  static_cast<uint32_t*>(outputs[0]->data)[i] = aclDataTypeTraits<ACL_UINT32>::from_float(val); break;
                    case ACL_BOOL:    static_cast<bool*>(outputs[0]->data)[i]     = aclDataTypeTraits<ACL_BOOL>::from_float(val); break;
                    default: break;
                }
                break;
            }
        }
    };

    for (size_t i = 0; i < count; ++i) {
        double d = readAsDouble(i);
        writeFromDouble(i, d);
    }
    return H_OK;
});

REGISTER_OP(LogicalAnd, {
    if (numInputs != 2 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[1] || !inputDesc[1] || !inputs[1]->data)
        return H_UNASSERTED;

    // LogicalAnd в torch_npu вызывается только для bool
    if (aclGetTensorDescType(inputDesc[0], false) != ACL_BOOL ||
        aclGetTensorDescType(inputDesc[1], false) != ACL_BOOL)
        return H_UNIMPLEMENTED;

    TensorAccessor<bool> inA(inputs[0]->data, inputDesc[0]->dims);
    TensorAccessor<bool> inB(inputs[1]->data, inputDesc[1]->dims);
    TensorAccessor<bool> out(outputs[0]->data, outputDesc[0]->dims);
    broadcastCompareOp<ACL_BOOL>(out, inA, inputDesc[0]->dims,
                                 inB, inputDesc[1]->dims,
                                 std::logical_and<bool>{});
    return H_OK;
});

REGISTER_OP(MaskedFill, {
    if (numInputs != 3 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[1] || !inputDesc[1] || !inputs[1]->data ||
        aclGetTensorDescType(inputDesc[1], false) != ACL_BOOL)
        return H_UNASSERTED;
    if (!inputs[2] || !inputDesc[2] || !inputs[2]->data)
        return H_UNASSERTED;

    aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
    switch (inDt) {
        DISPATCH_MASKED_FILL(ACL_FLOAT)
        DISPATCH_MASKED_FILL(ACL_DOUBLE)
        DISPATCH_MASKED_FILL(ACL_FLOAT16)
        DISPATCH_MASKED_FILL(ACL_BF16)
        DISPATCH_MASKED_FILL(ACL_INT8)
        DISPATCH_MASKED_FILL(ACL_UINT8)
        DISPATCH_MASKED_FILL(ACL_INT16)
        DISPATCH_MASKED_FILL(ACL_UINT16)
        DISPATCH_MASKED_FILL(ACL_INT32)
        DISPATCH_MASKED_FILL(ACL_UINT32)
        DISPATCH_MASKED_FILL(ACL_INT64)
        DISPATCH_MASKED_FILL(ACL_UINT64)
        DISPATCH_MASKED_FILL(ACL_BOOL)
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(MaskedSelect, {
    if (numInputs != 2 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[1] || !inputDesc[1] || !inputs[1]->data)
        return H_UNASSERTED;

    // маска всегда bool
    if (aclGetTensorDescType(inputDesc[1], false) != ACL_BOOL)
        return H_UNIMPLEMENTED;

    aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
    switch (inDt) {
        DISPATCH_MASKED_SELECT(ACL_FLOAT)
        DISPATCH_MASKED_SELECT(ACL_DOUBLE)
        DISPATCH_MASKED_SELECT(ACL_FLOAT16)
        DISPATCH_MASKED_SELECT(ACL_BF16)
        DISPATCH_MASKED_SELECT(ACL_INT8)
        DISPATCH_MASKED_SELECT(ACL_UINT8)
        DISPATCH_MASKED_SELECT(ACL_INT16)
        DISPATCH_MASKED_SELECT(ACL_UINT16)
        DISPATCH_MASKED_SELECT(ACL_INT32)
        DISPATCH_MASKED_SELECT(ACL_UINT32)
        DISPATCH_MASKED_SELECT(ACL_INT64)
        DISPATCH_MASKED_SELECT(ACL_UINT64)
        DISPATCH_MASKED_SELECT(ACL_BOOL)
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(Abs, {
    if (numInputs != 1 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;

    aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
    switch (inDt) {
        DISPATCH_UNARY(ACL_FLOAT,   std::abs)
        DISPATCH_UNARY(ACL_DOUBLE,  std::abs)
        DISPATCH_UNARY(ACL_INT8,    std::abs)
        DISPATCH_UNARY(ACL_INT16,   std::abs)
        DISPATCH_UNARY(ACL_INT32,   std::abs)
        DISPATCH_UNARY(ACL_INT64,   std::abs)
        // беззнаковые: abs эквивалентно копированию
        DISPATCH_UNARY(ACL_UINT8,   [](uint8_t v) { return v; })
        DISPATCH_UNARY(ACL_UINT16,  [](uint16_t v) { return v; })
        DISPATCH_UNARY(ACL_UINT32,  [](uint32_t v) { return v; })
        DISPATCH_UNARY(ACL_UINT64,  [](uint64_t v) { return v; })
        DISPATCH_UNARY(ACL_FLOAT16, [](uint16_t v) { return float_to_half(std::abs(half_to_float(v))); })
        DISPATCH_UNARY(ACL_BF16,    [](uint16_t v) { return float_to_bf16(std::abs(bf16_to_float(v))); })
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(ReduceAll, {
    if (numInputs != 2 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    size_t num_axes = calc_num_elements(inputDesc[1], inputs[1]->size);
    if (!inputs[1] || !inputDesc[1] || (num_axes > 0 && !inputs[1]->data) ||
        aclGetTensorDescType(inputDesc[1], false) != ACL_INT64)
        return H_UNASSERTED;

    aclDataType dt = aclGetTensorDescType(inputDesc[0], false);
    switch (dt) {
        DISPATCH_REDUCE_ALL(ACL_FLOAT)
        DISPATCH_REDUCE_ALL(ACL_DOUBLE)
        DISPATCH_REDUCE_ALL(ACL_FLOAT16)
        DISPATCH_REDUCE_ALL(ACL_BF16)
        DISPATCH_REDUCE_ALL(ACL_INT8)
        DISPATCH_REDUCE_ALL(ACL_UINT8)
        DISPATCH_REDUCE_ALL(ACL_INT16)
        DISPATCH_REDUCE_ALL(ACL_UINT16)
        DISPATCH_REDUCE_ALL(ACL_INT32)
        DISPATCH_REDUCE_ALL(ACL_UINT32)
        DISPATCH_REDUCE_ALL(ACL_INT64)
        DISPATCH_REDUCE_ALL(ACL_UINT64)
        DISPATCH_REDUCE_ALL(ACL_BOOL)
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(ReduceAny, {
    if (numInputs != 2 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    size_t num_axes = calc_num_elements(inputDesc[1], inputs[1]->size);
    if (!inputs[1] || !inputDesc[1] || (num_axes > 0 && !inputs[1]->data) ||
        aclGetTensorDescType(inputDesc[1], false) != ACL_INT64)
        return H_UNASSERTED;

    aclDataType dt = aclGetTensorDescType(inputDesc[0], false);
    switch (dt) {
        DISPATCH_REDUCE_ANY(ACL_FLOAT)
        DISPATCH_REDUCE_ANY(ACL_DOUBLE)
        DISPATCH_REDUCE_ANY(ACL_FLOAT16)
        DISPATCH_REDUCE_ANY(ACL_BF16)
        DISPATCH_REDUCE_ANY(ACL_INT8)
        DISPATCH_REDUCE_ANY(ACL_UINT8)
        DISPATCH_REDUCE_ANY(ACL_INT16)
        DISPATCH_REDUCE_ANY(ACL_UINT16)
        DISPATCH_REDUCE_ANY(ACL_INT32)
        DISPATCH_REDUCE_ANY(ACL_UINT32)
        DISPATCH_REDUCE_ANY(ACL_INT64)
        DISPATCH_REDUCE_ANY(ACL_UINT64)
        DISPATCH_REDUCE_ANY(ACL_BOOL)
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(ReduceMin, {
    if (numInputs != 2 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    size_t num_axes = calc_num_elements(inputDesc[1], inputs[1]->size);
    if (!inputs[1] || !inputDesc[1] || (num_axes > 0 && !inputs[1]->data) ||
        aclGetTensorDescType(inputDesc[1], false) != ACL_INT64)
        return H_UNASSERTED;

    aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
    switch (inDt) {
        DISPATCH_REDUCE_MIN(ACL_FLOAT)
        DISPATCH_REDUCE_MIN(ACL_DOUBLE)
        DISPATCH_REDUCE_MIN(ACL_FLOAT16)
        DISPATCH_REDUCE_MIN(ACL_BF16)
        DISPATCH_REDUCE_MIN(ACL_INT8)
        DISPATCH_REDUCE_MIN(ACL_UINT8)
        DISPATCH_REDUCE_MIN(ACL_INT16)
        DISPATCH_REDUCE_MIN(ACL_UINT16)
        DISPATCH_REDUCE_MIN(ACL_INT32)
        DISPATCH_REDUCE_MIN(ACL_UINT32)
        DISPATCH_REDUCE_MIN(ACL_INT64)
        DISPATCH_REDUCE_MIN(ACL_UINT64)
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(ReduceMax, {
    if (numInputs != 2 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[1] || !inputDesc[1])
        return H_UNASSERTED;
    size_t num_axes = calc_num_elements(inputDesc[1], inputs[1]->size);
    if (!inputs[1] || !inputDesc[1] || (num_axes > 0 && !inputs[1]->data) ||
        aclGetTensorDescType(inputDesc[1], false) != ACL_INT64)
        return H_UNASSERTED;

    aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
    switch (inDt) {
        DISPATCH_REDUCE_MAX(ACL_FLOAT)
        DISPATCH_REDUCE_MAX(ACL_DOUBLE)
        DISPATCH_REDUCE_MAX(ACL_FLOAT16)
        DISPATCH_REDUCE_MAX(ACL_BF16)
        DISPATCH_REDUCE_MAX(ACL_INT8)
        DISPATCH_REDUCE_MAX(ACL_UINT8)
        DISPATCH_REDUCE_MAX(ACL_INT16)
        DISPATCH_REDUCE_MAX(ACL_UINT16)
        DISPATCH_REDUCE_MAX(ACL_INT32)
        DISPATCH_REDUCE_MAX(ACL_UINT32)
        DISPATCH_REDUCE_MAX(ACL_INT64)
        DISPATCH_REDUCE_MAX(ACL_UINT64)
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(ArgMaxWithValue, {
    if (numInputs != 1 || numOutputs != 2)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!outputs[1] || !outputDesc[1] || !outputs[1]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    if (!attr)
        return H_UNASSERTED;

    // Проверяем, что атрибут dimension присутствует
    if (attr->ints.find("dimension") == attr->ints.end())
        return H_UNASSERTED;

    aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
    switch (inDt) {
        DISPATCH_ARG_MAX_WITH_VALUE(ACL_FLOAT)
        DISPATCH_ARG_MAX_WITH_VALUE(ACL_DOUBLE)
        DISPATCH_ARG_MAX_WITH_VALUE(ACL_FLOAT16)
        DISPATCH_ARG_MAX_WITH_VALUE(ACL_BF16)
        DISPATCH_ARG_MAX_WITH_VALUE(ACL_INT8)
        DISPATCH_ARG_MAX_WITH_VALUE(ACL_UINT8)
        DISPATCH_ARG_MAX_WITH_VALUE(ACL_INT16)
        DISPATCH_ARG_MAX_WITH_VALUE(ACL_UINT16)
        DISPATCH_ARG_MAX_WITH_VALUE(ACL_INT32)
        DISPATCH_ARG_MAX_WITH_VALUE(ACL_UINT32)
        DISPATCH_ARG_MAX_WITH_VALUE(ACL_INT64)
        DISPATCH_ARG_MAX_WITH_VALUE(ACL_UINT64)
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(ArgMinWithValue, {
    if (numInputs != 1 || numOutputs != 2)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!outputs[1] || !outputDesc[1] || !outputs[1]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    if (!attr)
        return H_UNASSERTED;
    if (attr->ints.find("dimension") == attr->ints.end())
        return H_UNASSERTED;

    aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
    switch (inDt) {
        DISPATCH_ARG_MIN_WITH_VALUE(ACL_FLOAT)
        DISPATCH_ARG_MIN_WITH_VALUE(ACL_DOUBLE)
        DISPATCH_ARG_MIN_WITH_VALUE(ACL_FLOAT16)
        DISPATCH_ARG_MIN_WITH_VALUE(ACL_BF16)
        DISPATCH_ARG_MIN_WITH_VALUE(ACL_INT8)
        DISPATCH_ARG_MIN_WITH_VALUE(ACL_UINT8)
        DISPATCH_ARG_MIN_WITH_VALUE(ACL_INT16)
        DISPATCH_ARG_MIN_WITH_VALUE(ACL_UINT16)
        DISPATCH_ARG_MIN_WITH_VALUE(ACL_INT32)
        DISPATCH_ARG_MIN_WITH_VALUE(ACL_UINT32)
        DISPATCH_ARG_MIN_WITH_VALUE(ACL_INT64)
        DISPATCH_ARG_MIN_WITH_VALUE(ACL_UINT64)
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(ReduceSum, {
    if (numInputs != 2 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    size_t num_axes = calc_num_elements(inputDesc[1], inputs[1]->size);
    if (!inputs[1] || !inputDesc[1] || (num_axes > 0 && !inputs[1]->data) ||
        aclGetTensorDescType(inputDesc[1], false) != ACL_INT64)
        return H_UNASSERTED;

    aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
    // Для ReduceSum обычно выходной тип совпадает с входным (или приводится к float для целых?),
    // но пока поддерживаем основные числовые типы как есть.
    switch (inDt) {
        DISPATCH_REDUCE(ACL_FLOAT,   (float)0,   [](float a,   float b)   { return a + b; })
        DISPATCH_REDUCE(ACL_DOUBLE,  (double)0,  [](double a,  double b)  { return a + b; })
        DISPATCH_REDUCE(ACL_FLOAT16, (uint16_t)0, [](uint16_t a, uint16_t b) {
            return float_to_half(half_to_float(a) + half_to_float(b));
        })
        DISPATCH_REDUCE(ACL_BF16,    (uint16_t)0, [](uint16_t a, uint16_t b) {
            return float_to_bf16(bf16_to_float(a) + bf16_to_float(b));
        })
        DISPATCH_REDUCE(ACL_INT8,    (int8_t)0,   std::plus<int8_t>{})
        DISPATCH_REDUCE(ACL_UINT8,   (uint8_t)0,  std::plus<uint8_t>{})
        DISPATCH_REDUCE(ACL_INT16,   (int16_t)0,  std::plus<int16_t>{})
        DISPATCH_REDUCE(ACL_UINT16,  (uint16_t)0, std::plus<uint16_t>{})
        DISPATCH_REDUCE(ACL_INT32,   (int32_t)0,  std::plus<int32_t>{})
        DISPATCH_REDUCE(ACL_UINT32,  (uint32_t)0, std::plus<uint32_t>{})
        DISPATCH_REDUCE(ACL_INT64,   (int64_t)0,  std::plus<int64_t>{})
        DISPATCH_REDUCE(ACL_UINT64,  (uint64_t)0, std::plus<uint64_t>{})
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(ReduceProd, {
    if (numInputs != 2 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    size_t num_axes = calc_num_elements(inputDesc[1], inputs[1]->size);
    if (!inputs[1] || !inputDesc[1] || (num_axes > 0 && !inputs[1]->data) ||
        aclGetTensorDescType(inputDesc[1], false) != ACL_INT64)
        return H_UNASSERTED;

    aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
    switch (inDt) {
        DISPATCH_REDUCE(ACL_FLOAT,   1.0f,              std::multiplies<float>{})
        DISPATCH_REDUCE(ACL_DOUBLE,  1.0,               std::multiplies<double>{})
        DISPATCH_REDUCE(ACL_FLOAT16, float_to_half(1.0f), [](uint16_t a, uint16_t b) {
            return float_to_half(half_to_float(a) * half_to_float(b));
        })
        DISPATCH_REDUCE(ACL_BF16,    float_to_bf16(1.0f), [](uint16_t a, uint16_t b) {
            return float_to_bf16(bf16_to_float(a) * bf16_to_float(b));
        })
        DISPATCH_REDUCE(ACL_INT8,    static_cast<int8_t>(1),    std::multiplies<int8_t>{})
        DISPATCH_REDUCE(ACL_UINT8,   static_cast<uint8_t>(1),   std::multiplies<uint8_t>{})
        DISPATCH_REDUCE(ACL_INT16,   static_cast<int16_t>(1),   std::multiplies<int16_t>{})
        DISPATCH_REDUCE(ACL_UINT16,  static_cast<uint16_t>(1),  std::multiplies<uint16_t>{})
        DISPATCH_REDUCE(ACL_INT32,   static_cast<int32_t>(1),   std::multiplies<int32_t>{})
        DISPATCH_REDUCE(ACL_UINT32,  static_cast<uint32_t>(1),  std::multiplies<uint32_t>{})
        DISPATCH_REDUCE(ACL_INT64,   static_cast<int64_t>(1),   std::multiplies<int64_t>{})
        DISPATCH_REDUCE(ACL_UINT64,  static_cast<uint64_t>(1),  std::multiplies<uint64_t>{})
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(ReduceMean, {
    if (numInputs != 2 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    size_t num_axes = calc_num_elements(inputDesc[1], inputs[1]->size);
    if (!inputs[1] || !inputDesc[1] || (num_axes > 0 && !inputs[1]->data) ||
        aclGetTensorDescType(inputDesc[1], false) != ACL_INT64)
        return H_UNASSERTED;

    aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
    switch (inDt) {
        DISPATCH_REDUCE_MEAN(ACL_FLOAT)
        DISPATCH_REDUCE_MEAN(ACL_DOUBLE)
        DISPATCH_REDUCE_MEAN(ACL_FLOAT16)
        DISPATCH_REDUCE_MEAN(ACL_BF16)
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(ReduceMeanD, {
    if (numInputs != 1 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    if (!attr)
        return H_UNASSERTED;

    aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
    switch (inDt) {
        DISPATCH_REDUCE_MEAN_FROM_AXES(ACL_FLOAT)
        DISPATCH_REDUCE_MEAN_FROM_AXES(ACL_DOUBLE)
        DISPATCH_REDUCE_MEAN_FROM_AXES(ACL_FLOAT16)
        DISPATCH_REDUCE_MEAN_FROM_AXES(ACL_BF16)
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(ReduceLogSumExp, {
    if (numInputs != 2 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    size_t num_axes = calc_num_elements(inputDesc[1], inputs[1]->size);
    if (!inputs[1] || !inputDesc[1] || (num_axes > 0 && !inputs[1]->data) ||
        aclGetTensorDescType(inputDesc[1], false) != ACL_INT64)
        return H_UNASSERTED;

    aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
    switch (inDt) {
        DISPATCH_REDUCE_LOG_SUM_EXP(ACL_FLOAT)
        DISPATCH_REDUCE_LOG_SUM_EXP(ACL_DOUBLE)
        DISPATCH_REDUCE_LOG_SUM_EXP(ACL_FLOAT16)
        DISPATCH_REDUCE_LOG_SUM_EXP(ACL_BF16)
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(ReduceStdV2Update, {
    if (numInputs != 2 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[1] || !inputDesc[1] || !inputs[1]->data)
        return H_UNASSERTED;
    if (!attr)
        return H_UNASSERTED;  // атрибуты обязательны

    // Извлекаем атрибуты
    // dim – список осей (int64_t)
    const auto& dims_attr = attr->list_ints.find("dim");
    if (dims_attr == attr->list_ints.end() || dims_attr->second.empty())
        return H_UNASSERTED;  // оси не заданы
    const std::vector<int64_t>& dims = dims_attr->second;

    // unbiased – bool
    bool unbiased = true;
    auto unbiased_it = attr->bools.find("unbiased");
    if (unbiased_it != attr->bools.end())
        unbiased = unbiased_it->second;

    // correction – int64_t
    int64_t correction = 1;
    auto correction_it = attr->ints.find("correction");
    if (correction_it != attr->ints.end())
        correction = correction_it->second;

    // keepdim – bool (но выход уже имеет размерность без учёта keepdim или с ней? в эмуляции можно игнорировать, т.к. форма выхода уже задана)
    // if_std – bool (0 – дисперсия, 1 – станд.отклонение, но мы пока делаем дисперсию)

    // Входные тензоры: self и mean_broadcast
    aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
    aclDataType meanDt = aclGetTensorDescType(inputDesc[1], false);
    if (inDt != meanDt)
        return H_UNASSERTED;

    // Поддерживаем только float/double/float16/bf16 для дисперсии
    switch (inDt) {
        case ACL_FLOAT:
        case ACL_DOUBLE:
        case ACL_FLOAT16:
        case ACL_BF16:
            break;
        default:
            return H_UNIMPLEMENTED;
    }

    size_t total_elements = calc_num_elements(inputDesc[0], inputs[0]->size);
    size_t out_elements = calc_num_elements(outputDesc[0], outputs[0]->size);
    if (total_elements == 0 || out_elements == 0)
        return H_OK; // пустой тензор

    // Нормализуем оси (могут быть отрицательными)
    std::vector<int64_t> norm_dims = dims;
    int64_t ndim = static_cast<int64_t>(inputDesc[0]->dims.size());
    for (auto& d : norm_dims) {
        if (d < 0) d += ndim;
    }

    // Определяем количество элементов, сворачиваемых в один выход
    int64_t N = 1;
    for (auto d : norm_dims) {
        if (d >= 0 && d < ndim)
            N *= inputDesc[0]->dims[d];
    }
    // Если оси не заданы (редукция по всем осям), N = total_elements
    if (norm_dims.empty())
        N = total_elements;

    // Вычисляем знаменатель
    double denom = static_cast<double>(N - correction);
    if (denom <= 0)
        denom = 0; // или NAN, но пока так

    switch (inDt) {
        REDUCE_STD_DT(ACL_FLOAT)
        REDUCE_STD_DT(ACL_DOUBLE)
        REDUCE_STD_DT(ACL_FLOAT16)
        REDUCE_STD_DT(ACL_BF16)
        default: break;
    }

    return H_OK;
});

REGISTER_OP(Ceil, {
    if (numInputs != 1 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;

    aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
    switch (inDt) {
        DISPATCH_UNARY(ACL_FLOAT,   std::ceil)
        DISPATCH_UNARY(ACL_DOUBLE,  std::ceil)
        DISPATCH_UNARY(ACL_FLOAT16, [](uint16_t v) { return float_to_half(std::ceil(half_to_float(v))); })
        DISPATCH_UNARY(ACL_BF16,    [](uint16_t v) { return float_to_bf16(std::ceil(bf16_to_float(v))); })
        // Целые типы не должны попадать в Ceil на NPU (torch_npu делает проверку ceil_integral_identity)
        // Если всё же попали — возвращаем H_UNIMPLEMENTED
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(Eye, {
    // Eye не имеет входных тензоров, только выход
    if (numInputs != 0 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    // attr не обязателен, так как размеры и dtype уже есть в выходном дескрипторе

    size_t ndim = aclGetTensorDescNumDims(outputDesc[0], false);
    if (ndim != 2)
        return H_UNASSERTED;   // Eye всегда создаёт двумерную матрицу

    int64_t n = 0; int64_t m = 0; // А это забавно: использование запятой РАЗРУШАЕТ МАКРОСЫ!) Оно путает это с межаргументной запятой
    if (aclGetTensorDescDimV2(outputDesc[0], 0, &n, false) != ACL_SUCCESS ||
        aclGetTensorDescDimV2(outputDesc[0], 1, &m, false) != ACL_SUCCESS)
        return H_UNASSERTED;

    aclDataType dt = aclGetTensorDescType(outputDesc[0], false);
    size_t elemSize = aclDataTypeBytes(dt);
    char* data = static_cast<char*>(outputs[0]->data);

    // Заполняем всё нулями
    std::memset(data, 0, n * m * elemSize);

    // Устанавливаем единицы на главной диагонали
    for (int64_t i = 0; i < std::min(n, m); ++i) {
        size_t idx = i * m + i;                     // C-order (row-major)
        char* elemPtr = data + idx * elemSize;
        switch (dt) {
            case ACL_FLOAT:   *reinterpret_cast<float*>(elemPtr) = 1.0f; break;
            case ACL_DOUBLE:  *reinterpret_cast<double*>(elemPtr) = 1.0; break;
            case ACL_INT8:    *reinterpret_cast<int8_t*>(elemPtr) = 1; break;
            case ACL_UINT8:   *reinterpret_cast<uint8_t*>(elemPtr) = 1; break;
            case ACL_INT16:   *reinterpret_cast<int16_t*>(elemPtr) = 1; break;
            case ACL_UINT16:  *reinterpret_cast<uint16_t*>(elemPtr) = 1; break;
            case ACL_INT32:   *reinterpret_cast<int32_t*>(elemPtr) = 1; break;
            case ACL_UINT32:  *reinterpret_cast<uint32_t*>(elemPtr) = 1; break;
            case ACL_INT64:   *reinterpret_cast<int64_t*>(elemPtr) = 1; break;
            case ACL_UINT64:  *reinterpret_cast<uint64_t*>(elemPtr) = 1; break;
            case ACL_BOOL:    *reinterpret_cast<bool*>(elemPtr) = true; break;
            case ACL_FLOAT16: *reinterpret_cast<uint16_t*>(elemPtr) = float_to_half(1.0f); break;
            case ACL_BF16:    *reinterpret_cast<uint16_t*>(elemPtr) = float_to_bf16(1.0f); break;
            default:
                return H_UNIMPLEMENTED;
        }
    }
    return H_OK;
});

REGISTER_OP(BroadcastTo, {
    if (numInputs != 2 || numOutputs != 1)
        return H_UNASSERTED;
    if (!outputs[0] || !outputDesc[0] || !outputs[0]->data)
        return H_UNASSERTED;
    if (!inputs[0] || !inputDesc[0] || !inputs[0]->data)
        return H_UNASSERTED;
    // inputs[1] — целевая форма, не используется в эмуляции

    aclDataType dt = aclGetTensorDescType(outputDesc[0], false);
    switch (dt) {
        DISPATCH_BROADCAST_TO(ACL_FLOAT)
        DISPATCH_BROADCAST_TO(ACL_DOUBLE)
        DISPATCH_BROADCAST_TO(ACL_FLOAT16)
        DISPATCH_BROADCAST_TO(ACL_BF16)
        DISPATCH_BROADCAST_TO(ACL_INT8)
        DISPATCH_BROADCAST_TO(ACL_UINT8)
        DISPATCH_BROADCAST_TO(ACL_INT16)
        DISPATCH_BROADCAST_TO(ACL_UINT16)
        DISPATCH_BROADCAST_TO(ACL_INT32)
        DISPATCH_BROADCAST_TO(ACL_UINT32)
        DISPATCH_BROADCAST_TO(ACL_INT64)
        DISPATCH_BROADCAST_TO(ACL_UINT64)
        DISPATCH_BROADCAST_TO(ACL_BOOL)
        default:
            return H_UNIMPLEMENTED;
    }
    return H_OK;
});

REGISTER_OP(ConcatD, {
    int N;
    int64_t concat_dim;
    std::vector<at::Tensor> tensors;
    at::Tensor out_tensor;

    ASSERT(numInputs >= 1 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(attr)
    ASSERT(try_get_attr<int>(attr, "N", N) && N == numInputs)
    ASSERT(try_get_attr<int64_t>(attr, "concat_dim", concat_dim))

    TRY(toAtenTensors(N, inputDesc, inputs, tensors));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out_tensor));

    out_tensor.copy_(at::cat(tensors, concat_dim));
    return H_OK;
});

REGISTER_OP(Dummy, {
    return H_OK;
});



ACL_FUNC_VISIBILITY aclError aclopCompileAndExecute(const char *opType,
    int numInputs, const aclTensorDesc *const inputDesc[], const aclDataBuffer *const inputs[],
    int numOutputs, const aclTensorDesc *const outputDesc[], aclDataBuffer *const outputs[],
    const aclopAttr *attr, aclopEngineType engineType, aclopCompileType compileFlag,
    const char *opPath, aclrtStream stream) {

    std::ostringstream log;
    log << "[aclopCompileAndExecute] opType=" << (opType ? opType : "(null)")
        << " opPath=" << (opPath ? opPath : "(null)") << " stream=" << stream
        << "\n    numInputs=" << numInputs << " numOutputs=" << numOutputs
    << formatTensorList("input", inputDesc, inputs, numInputs, PRINT_ALL)
    << formatTensorList("output", outputDesc, outputs, numOutputs, PRINT_DESC);

    OpHandler handler;
    exitCode code = H_UNKNOWN_OP;
    if (opType && OpRegistry::try_find(opType, handler))
        code = handler(numInputs, inputDesc, inputs, numOutputs, outputDesc, outputs, attr);

    switch (code) {
        case H_UNKNOWN_OP:
            log << "\nError: unknown operation: " << (opType ? opType : "(null)");
            log_output(log, true);
            return ACL_ERROR_OP_NOT_FOUND;
        case H_OK:
            break;
        case H_UNASSERTED:
            log << "\nError: assertion failed for operation " << opType;
            log_output(log, true);
            return ACL_ERROR_INVALID_PARAM;
        case H_UNIMPLEMENTED:
            log << "\nError: unsupported dtype for operation " << opType;
            log_output(log, true);
            return ACL_ERROR_UNSUPPORTED_DATA_TYPE;
    }

    log << formatTensorList("output", outputDesc, outputs, numOutputs, PRINT_DATA);
    log_output(log);

    return ACL_SUCCESS;
}

// ========== aclopCompileAndExecuteV2 делегирует первой версии ==========
ACL_FUNC_VISIBILITY aclError aclopCompileAndExecuteV2(const char *opType,
    int numInputs, aclTensorDesc *inputDesc[], aclDataBuffer *inputs[],
    int numOutputs, aclTensorDesc *outputDesc[], aclDataBuffer *outputs[],
    aclopAttr *attr, aclopEngineType engineType, aclopCompileType compileFlag,
    const char *opPath, aclrtStream stream) {

    log_output("[aclopCompileAndExecuteV2] delegating to aclopCompileAndExecute");

    return aclopCompileAndExecute(opType,
        numInputs, (const aclTensorDesc* const*)inputDesc, (const aclDataBuffer* const*)inputs,
        numOutputs, (const aclTensorDesc* const*)outputDesc, outputs,
        attr, engineType, compileFlag, opPath, stream);
}


#ifdef __cplusplus
}
#endif
