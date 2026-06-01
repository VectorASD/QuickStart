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



ACL_FUNC_VISIBILITY aclError aclopCompileAndExecute(const char *opType,
    int numInputs, const aclTensorDesc *const inputDesc[], const aclDataBuffer *const inputs[],
    int numOutputs, const aclTensorDesc *const outputDesc[], aclDataBuffer *const outputs[],
    const aclopAttr *attr, aclopEngineType engineType, aclopCompileType compileFlag,
    const char *opPath, aclrtStream stream) {

    std::ostringstream log;
    log << "[aclopCompileAndExecute] opType=" << (opType ? opType : "(null)")
        << " opPath=" << (opPath ? opPath : "(null)")
        << "\n    numInputs=" << numInputs << " numOutputs=" << numOutputs << '\n'
        << formatTensorList("input", inputDesc, inputs, numInputs);
 // log_output(log);

    // --- эмуляция операций ---
    if (!opType) {
    } else if (strstr(opType, "StatelessRandomNormalV2")) {
        // Извлечение seed (остаётся без изменений)
        uint64_t seed_value = 0;
        bool has_seed = false;
        if (numInputs >= 2 && inputs[1] && inputDesc[1] && inputs[1]->data) {
            if (aclGetTensorDescType(inputDesc[1], false) == ACL_UINT64) {
                size_t seed_count = calc_num_elements(inputDesc[1], inputs[1]->size);
                if (seed_count >= 1) {
                    const uint64_t* seed_ptr = static_cast<const uint64_t*>(inputs[1]->data);
                    seed_value = seed_ptr[0];
                    has_seed = true;
                }
            }
        }

        std::mt19937 local_rng;
        if (has_seed) {
            local_rng.seed(static_cast<std::mt19937::result_type>(seed_value));
        } else {
            std::random_device rd;
            local_rng.seed(rd());
        }

        for (int i = 0; i < numOutputs; ++i) {
            if (outputs[i] && outputDesc[i] && outputs[i]->data) {
                aclDataType dt = aclGetTensorDescType(outputDesc[i], false);
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
                // Остальные типы не поддерживаются, fallback — оставляем буфер без изменений
                default: break;
                }
            }
        }
    } else if (strcmp(opType, "ZerosLike") == 0 && numInputs == 1 && numOutputs == 1) {
        aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
        if (outputs[0] && outputs[0]->data) {
            switch (inDt) {
                DISPATCH_ZEROS_LIKE(ACL_FLOAT)
                DISPATCH_ZEROS_LIKE(ACL_DOUBLE)
                DISPATCH_ZEROS_LIKE(ACL_INT8)
                DISPATCH_ZEROS_LIKE(ACL_UINT8)
                DISPATCH_ZEROS_LIKE(ACL_INT16)
                DISPATCH_ZEROS_LIKE(ACL_UINT16)
                DISPATCH_ZEROS_LIKE(ACL_INT32)
                DISPATCH_ZEROS_LIKE(ACL_UINT32)
                DISPATCH_ZEROS_LIKE(ACL_INT64)
                DISPATCH_ZEROS_LIKE(ACL_UINT64)
                DISPATCH_ZEROS_LIKE(ACL_FLOAT16)
                DISPATCH_ZEROS_LIKE(ACL_BF16)
                DISPATCH_ZEROS_LIKE(ACL_BOOL)
            default:
                size_t count = calc_num_elements(outputDesc[0], outputs[0]->size);
                size_t elemSize = aclDataTypeBytes(inDt);
                if (count > 0 && elemSize > 0) {
                    memset(outputs[0]->data, 0, count * elemSize);
                }
                break;
            }
        }
    } else if (strcmp(opType, "Fill") == 0 && numInputs == 2 && numOutputs == 1) {
        aclDataType outDt = aclGetTensorDescType(outputDesc[0], false);
        if (outputs[0] && outputs[0]->data && inputs[1] && inputs[1]->data) {
            switch (outDt) {
                DISPATCH_FILL(ACL_FLOAT)
                DISPATCH_FILL(ACL_DOUBLE)
                DISPATCH_FILL(ACL_INT8)
                DISPATCH_FILL(ACL_UINT8)
                DISPATCH_FILL(ACL_INT16)
                DISPATCH_FILL(ACL_UINT16)
                DISPATCH_FILL(ACL_INT32)
                DISPATCH_FILL(ACL_UINT32)
                DISPATCH_FILL(ACL_INT64)
                DISPATCH_FILL(ACL_UINT64)
                DISPATCH_FILL(ACL_FLOAT16)
                DISPATCH_FILL(ACL_BF16)
                DISPATCH_FILL(ACL_BOOL)
            default:
                // fallback: копируем первый байт из значения во весь выход (если размеры совпадают)
                size_t count = calc_num_elements(outputDesc[0], outputs[0]->size);
                size_t elemSize = aclDataTypeBytes(outDt);
                if (count > 0 && elemSize > 0) {
                    // предполагаем, что inputs[1]->size >= elemSize
                    for (size_t i = 0; i < count; ++i) {
                        memcpy(static_cast<char*>(outputs[0]->data) + i * elemSize, inputs[1]->data, elemSize);
                    }
                }
                break;
            }
        }
    } else if (strcmp(opType, "Mul") == 0 && numInputs == 2 && numOutputs == 1) {
        // Поэлементное умножение с broadcasting
        aclDataType dt = aclGetTensorDescType(outputDesc[0], false);
        if (outputs[0] && outputs[0]->data && inputs[0] && inputs[0]->data && inputs[1] && inputs[1]->data) {
            switch (dt) {
                DISPATCH_MUL(ACL_FLOAT)
                DISPATCH_MUL(ACL_DOUBLE)
                DISPATCH_MUL(ACL_INT8)
                DISPATCH_MUL(ACL_UINT8)
                DISPATCH_MUL(ACL_INT16)
                DISPATCH_MUL(ACL_UINT16)
                DISPATCH_MUL(ACL_INT32)
                DISPATCH_MUL(ACL_UINT32)
                DISPATCH_MUL(ACL_INT64)
                DISPATCH_MUL(ACL_UINT64)
                DISPATCH_MUL(ACL_FLOAT16)
                DISPATCH_MUL(ACL_BF16)
            default:
                // fallback: копируем первый вход в выход (для неподдерживаемых типов)
                size_t count = calc_num_elements(outputDesc[0], outputs[0]->size);
                size_t elemSize = aclDataTypeBytes(dt);
                if (count > 0 && elemSize > 0) {
                    memcpy(outputs[0]->data, inputs[0]->data, std::min(count * elemSize, outputs[0]->size));
                }
                break;
            }
        }
    } else if (strcmp(opType, "Add") == 0 && numInputs == 2 && numOutputs == 1) {
        // Поэлементное сложение с broadcasting
        aclDataType dt = aclGetTensorDescType(outputDesc[0], false);
        if (outputs[0] && outputs[0]->data && inputs[0] && inputs[0]->data && inputs[1] && inputs[1]->data) {
            switch (dt) {
                DISPATCH_ADD(ACL_FLOAT)
                DISPATCH_ADD(ACL_DOUBLE)
                DISPATCH_ADD(ACL_INT8)
                DISPATCH_ADD(ACL_UINT8)
                DISPATCH_ADD(ACL_INT16)
                DISPATCH_ADD(ACL_UINT16)
                DISPATCH_ADD(ACL_INT32)
                DISPATCH_ADD(ACL_UINT32)
                DISPATCH_ADD(ACL_INT64)
                DISPATCH_ADD(ACL_UINT64)
                DISPATCH_ADD(ACL_FLOAT16)
                DISPATCH_ADD(ACL_BF16)
            default:
                size_t count = calc_num_elements(outputDesc[0], outputs[0]->size);
                size_t elemSize = aclDataTypeBytes(dt);
                if (count > 0 && elemSize > 0) {
                    memcpy(outputs[0]->data, inputs[0]->data, std::min(count * elemSize, outputs[0]->size));
                }
                break;
            }
        }
    } else if (strcmp(opType, "RealDiv") == 0 && numInputs == 2 && numOutputs == 1) {
        aclDataType dt = aclGetTensorDescType(outputDesc[0], false);
        if (outputs[0] && outputs[0]->data && inputs[0] && inputs[0]->data && inputs[1] && inputs[1]->data) {
            switch (dt) {
                DISPATCH_DIV(ACL_FLOAT)
                DISPATCH_DIV(ACL_DOUBLE)
                DISPATCH_DIV(ACL_INT8)    // деление целых — осторожно!
                DISPATCH_DIV(ACL_UINT8)
                DISPATCH_DIV(ACL_INT16)
                DISPATCH_DIV(ACL_UINT16)
                DISPATCH_DIV(ACL_INT32)
                DISPATCH_DIV(ACL_UINT32)
                DISPATCH_DIV(ACL_INT64)
                DISPATCH_DIV(ACL_UINT64)
                DISPATCH_DIV(ACL_FLOAT16)
                DISPATCH_DIV(ACL_BF16)
            default:
                size_t count = calc_num_elements(outputDesc[0], outputs[0]->size);
                size_t elemSize = aclDataTypeBytes(dt);
                if (count > 0 && elemSize > 0) {
                    memcpy(outputs[0]->data, inputs[0]->data, std::min(count * elemSize, outputs[0]->size));
                }
                break;
            }
        }
    } else if (strcmp(opType, "IsFinite") == 0 && numInputs == 1 && numOutputs == 1) {
        aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
        if (outputs[0] && outputs[0]->data && inputs[0] && inputs[0]->data) {
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
                // fallback: заполняем false
                size_t count = calc_num_elements(outputDesc[0], outputs[0]->size);
                if (count > 0 && outputs[0]->data) {
                    memset(outputs[0]->data, 0, count * sizeof(bool));
                }
                break;
            }
        }
    } else if (strcmp(opType, "NotEqual") == 0 && numInputs == 2 && numOutputs == 1) {
        aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
        if (outputs[0] && outputs[0]->data && inputs[0] && inputs[0]->data && inputs[1] && inputs[1]->data) {
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
                // Для half/bf16: преобразуем в float внутри предиката, или используем собственный функтор
                DISPATCH_COMPARE(ACL_FLOAT16,  [](uint16_t a, uint16_t b) { return half_to_float(a) != half_to_float(b); })
                DISPATCH_COMPARE(ACL_BF16,     [](uint16_t a, uint16_t b) { return bf16_to_float(a) != bf16_to_float(b); })
                DISPATCH_COMPARE(ACL_BOOL,     std::not_equal_to<bool>{})
            default:
                size_t count = calc_num_elements(outputDesc[0], outputs[0]->size);
                if (count > 0 && outputs[0]->data) {
                    memset(outputs[0]->data, 0, count * sizeof(bool));
                }
                break;
            }
        }
    } else if (strcmp(opType, "LogicalAnd") == 0 && numInputs == 2 && numOutputs == 1) {
        aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
        if (outputs[0] && outputs[0]->data && inputs[0] && inputs[0]->data && inputs[1] && inputs[1]->data) {
            switch (inDt) {
                DISPATCH_LOGICAL(ACL_BOOL, std::logical_and<bool>{})
            default:
                size_t count = calc_num_elements(outputDesc[0], outputs[0]->size);
                if (count > 0 && outputs[0]->data) {
                    memset(outputs[0]->data, 0, count * sizeof(bool));
                }
                break;
            }
        }
    } else if (strcmp(opType, "MaskedSelect") == 0 && numInputs == 2 && numOutputs == 1) {
        aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
        if (outputs[0] && outputs[0]->data && inputs[0] && inputs[0]->data && inputs[1] && inputs[1]->data) {
            switch (inDt) {
                DISPATCH_MASKED_SELECT(ACL_FLOAT)
                DISPATCH_MASKED_SELECT(ACL_DOUBLE)
                DISPATCH_MASKED_SELECT(ACL_INT8)
                DISPATCH_MASKED_SELECT(ACL_UINT8)
                DISPATCH_MASKED_SELECT(ACL_INT16)
                DISPATCH_MASKED_SELECT(ACL_UINT16)
                DISPATCH_MASKED_SELECT(ACL_INT32)
                DISPATCH_MASKED_SELECT(ACL_UINT32)
                DISPATCH_MASKED_SELECT(ACL_INT64)
                DISPATCH_MASKED_SELECT(ACL_UINT64)
                DISPATCH_MASKED_SELECT(ACL_FLOAT16)
                DISPATCH_MASKED_SELECT(ACL_BF16)
                DISPATCH_MASKED_SELECT(ACL_BOOL)
            default:
                break; // оставляем выходной буфер как есть
            }
        }
    } else if (strcmp(opType, "Abs") == 0 && numInputs == 1 && numOutputs == 1) {
        aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);  // false подавляет лог
        if (outputs[0] && outputs[0]->data && inputs[0] && inputs[0]->data) {
            switch (inDt) {
                DISPATCH_UNARY(ACL_FLOAT,   std::abs)
                DISPATCH_UNARY(ACL_DOUBLE,  std::abs)
                DISPATCH_UNARY(ACL_INT8,    std::abs)
                DISPATCH_UNARY(ACL_INT16,   std::abs)
                DISPATCH_UNARY(ACL_INT32,   std::abs)
                DISPATCH_UNARY(ACL_INT64,   std::abs)
                DISPATCH_UNARY(ACL_FLOAT16, [](uint16_t v) { return float_to_half(std::abs(half_to_float(v))); })
                DISPATCH_UNARY(ACL_BF16,    [](uint16_t v) { return float_to_bf16(std::abs(bf16_to_float(v))); })
            default:
                // fallback: копируем вход в выход
                size_t count = calc_num_elements(outputDesc[0], outputs[0]->size);
                size_t elemSize = aclDataTypeBytes(inDt);
                if (count > 0 && elemSize > 0) {
                    memcpy(outputs[0]->data, inputs[0]->data, count * elemSize);
                }
                break;
            }
        }
    } else if (strcmp(opType, "ReduceMin") == 0 && numInputs == 2 && numOutputs == 1) {
        aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
        if (outputs[0] && outputs[0]->data && inputs[0] && inputs[0]->data) {
            switch (inDt) {
                DISPATCH_REDUCE_MIN(ACL_FLOAT)
                DISPATCH_REDUCE_MIN(ACL_DOUBLE)
                DISPATCH_REDUCE_MIN(ACL_INT8)
                DISPATCH_REDUCE_MIN(ACL_UINT8)
                DISPATCH_REDUCE_MIN(ACL_INT16)
                DISPATCH_REDUCE_MIN(ACL_UINT16)
                DISPATCH_REDUCE_MIN(ACL_INT32)
                DISPATCH_REDUCE_MIN(ACL_UINT32)
                DISPATCH_REDUCE_MIN(ACL_INT64)
                DISPATCH_REDUCE_MIN(ACL_UINT64)
                DISPATCH_REDUCE_MIN(ACL_FLOAT16)
                DISPATCH_REDUCE_MIN(ACL_BF16)
            default:
                // fallback: копируем первый элемент входа в выходной буфер
                size_t elemSize = aclDataTypeBytes(inDt);
                size_t outCount = calc_num_elements(outputDesc[0], outputs[0]->size);
                if (outCount > 0 && elemSize > 0 && outputs[0]->data && inputs[0]->data) {
                    memcpy(outputs[0]->data, inputs[0]->data, std::min(elemSize, outputs[0]->size));
                }
                break;
            }
        }
    } else if (strcmp(opType, "ReduceMax") == 0 && numInputs == 2 && numOutputs == 1) {
        aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
        if (outputs[0] && outputs[0]->data && inputs[0] && inputs[0]->data && inputs[1] && inputs[1]->data) {
            switch (inDt) {
                DISPATCH_REDUCE_MAX(ACL_FLOAT)
                DISPATCH_REDUCE_MAX(ACL_DOUBLE)
                DISPATCH_REDUCE_MAX(ACL_INT8)
                DISPATCH_REDUCE_MAX(ACL_UINT8)
                DISPATCH_REDUCE_MAX(ACL_INT16)
                DISPATCH_REDUCE_MAX(ACL_UINT16)
                DISPATCH_REDUCE_MAX(ACL_INT32)
                DISPATCH_REDUCE_MAX(ACL_UINT32)
                DISPATCH_REDUCE_MAX(ACL_INT64)
                DISPATCH_REDUCE_MAX(ACL_UINT64)
                DISPATCH_REDUCE_MAX(ACL_FLOAT16)
                DISPATCH_REDUCE_MAX(ACL_BF16)
            default:
                // fallback: копируем первый элемент входа в выходной буфер
                size_t elemSize = aclDataTypeBytes(inDt);
                size_t outCount = calc_num_elements(outputDesc[0], outputs[0]->size);
                if (outCount > 0 && elemSize > 0 && outputs[0]->data && inputs[0]->data) {
                    memcpy(outputs[0]->data, inputs[0]->data, std::min(elemSize, outputs[0]->size));
                }
                break;
            }
        }
    } else if (strcmp(opType, "Ceil") == 0 && numInputs == 1 && numOutputs == 1) {
        aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
        if (outputs[0] && outputs[0]->data && inputs[0] && inputs[0]->data) {
            switch (inDt) {
                DISPATCH_UNARY(ACL_FLOAT,   std::ceil)
                DISPATCH_UNARY(ACL_DOUBLE,  std::ceil)
                DISPATCH_UNARY(ACL_FLOAT16, [](uint16_t v) { return float_to_half(std::ceil(half_to_float(v))); })
                DISPATCH_UNARY(ACL_BF16,    [](uint16_t v) { return float_to_bf16(std::ceil(bf16_to_float(v))); })
            default:
                // Для целых типов ceil не меняет значение, просто копируем вход
                size_t count = calc_num_elements(outputDesc[0], outputs[0]->size);
                size_t elemSize = aclDataTypeBytes(inDt);
                if (count > 0 && elemSize > 0) {
                    memcpy(outputs[0]->data, inputs[0]->data, std::min(count * elemSize, outputs[0]->size));
                }
                break;
            }
        }
    } else if (strcmp(opType, "Greater") == 0 && numInputs == 2 && numOutputs == 1) {
        aclDataType inDt = aclGetTensorDescType(inputDesc[0], false);
        if (outputs[0] && outputs[0]->data && inputs[0] && inputs[0]->data && inputs[1] && inputs[1]->data) {
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
                size_t count = calc_num_elements(outputDesc[0], outputs[0]->size);
                if (count > 0 && outputs[0]->data) {
                    memset(outputs[0]->data, 0, count * sizeof(bool));
                }
                break;
            }
        }
    }

    log << formatTensorList("output", outputDesc, outputs, numOutputs)
        << "    stream=" << stream;
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
