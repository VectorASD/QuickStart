#include "common.h"
#include "not_acl.cpp"  // aclGetTensorDescType, calc_num_elements
#include <cstring>      // memset, size_t
#include <iostream>     // cout, endl
#include <random>       // bernoulli_distribution, mt19937, normal_distribution, random_device, uniform_int_distribution

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
        case ACL_PRECISION_MODE:            return "ACL_PRECISION_MODE";
        case ACL_AICORE_NUM:                return "ACL_AICORE_NUM";
        case ACL_AUTO_TUNE_MODE:            return "ACL_AUTO_TUNE_MODE";
        case ACL_OP_SELECT_IMPL_MODE:       return "ACL_OP_SELECT_IMPL_MODE";
        case ACL_OPTYPELIST_FOR_IMPLMODE:   return "ACL_OPTYPELIST_FOR_IMPLMODE";
        case ACL_OP_DEBUG_LEVEL:            return "ACL_OP_DEBUG_LEVEL";
        case ACL_DEBUG_DIR:                 return "ACL_DEBUG_DIR";
        case ACL_OP_COMPILER_CACHE_MODE:    return "ACL_OP_COMPILER_CACHE_MODE";
        case ACL_OP_COMPILER_CACHE_DIR:     return "ACL_OP_COMPILER_CACHE_DIR";
        case ACL_OP_PERFORMANCE_MODE:       return "ACL_OP_PERFORMANCE_MODE";
        case ACL_OP_JIT_COMPILE:            return "ACL_OP_JIT_COMPILE";
        case ACL_OP_DETERMINISTIC:          return "ACL_OP_DETERMINISTIC";
        case ACL_CUSTOMIZE_DTYPES:           return "ACL_CUSTOMIZE_DTYPES";
        case ACL_OP_PRECISION_MODE:         return "ACL_OP_PRECISION_MODE";
        case ACL_ALLOW_HF32:                return "ACL_ALLOW_HF32";
        case ACL_PRECISION_MODE_V2:         return "ACL_PRECISION_MODE_V2";
        case ACL_OP_DEBUG_OPTION:           return "ACL_OP_DEBUG_OPTION";
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


static std::mt19937 g_rng(std::random_device{}());

static void fill_random_normal(aclDataType dtype, void* data, size_t count) {
    if (count == 0 || !data) return;
    // нормальное распределение N(0,1)
    switch (dtype) {
        case ACL_FLOAT: {
            auto* p = static_cast<float*>(data);
            std::normal_distribution<float> dist(0.0f, 1.0f);
            for (size_t i = 0; i < count; ++i) p[i] = dist(g_rng);
            break;
        }
        case ACL_FLOAT16:
        case ACL_BF16: {
            // для half/bfloat16 генерируем float и преобразуем приближённо
            auto* p = static_cast<uint16_t*>(data);
            std::normal_distribution<float> dist(0.0f, 1.0f);
            for (size_t i = 0; i < count; ++i) {
                float v = dist(g_rng);
                // грубое преобразование float -> half (берём старшие 16 бит float)
                uint32_t bits = reinterpret_cast<uint32_t&>(v);
                p[i] = static_cast<uint16_t>(bits >> 16);
            }
            break;
        }
        case ACL_DOUBLE: {
            auto* p = static_cast<double*>(data);
            std::normal_distribution<double> dist(0.0, 1.0);
            for (size_t i = 0; i < count; ++i) p[i] = dist(g_rng);
            break;
        }
        // для целых типов – равномерное распределение (условно)
        case ACL_INT8: {
            auto* p = static_cast<int8_t*>(data);
            std::uniform_int_distribution<int16_t> dist(-128, 127);
            for (size_t i = 0; i < count; ++i) p[i] = static_cast<int8_t>(dist(g_rng));
            break;
        }
        case ACL_INT16: {
            auto* p = static_cast<int16_t*>(data);
            std::uniform_int_distribution<int16_t> dist;
            for (size_t i = 0; i < count; ++i) p[i] = dist(g_rng);
            break;
        }
        case ACL_INT32: {
            auto* p = static_cast<int32_t*>(data);
            std::uniform_int_distribution<int32_t> dist;
            for (size_t i = 0; i < count; ++i) p[i] = dist(g_rng);
            break;
        }
        case ACL_INT64: {
            auto* p = static_cast<int64_t*>(data);
            std::uniform_int_distribution<int64_t> dist;
            for (size_t i = 0; i < count; ++i) p[i] = dist(g_rng);
            break;
        }
        case ACL_UINT8: {
            auto* p = static_cast<uint8_t*>(data);
            std::uniform_int_distribution<uint16_t> dist(0, 255);
            for (size_t i = 0; i < count; ++i) p[i] = static_cast<uint8_t>(dist(g_rng));
            break;
        }
        case ACL_UINT16: {
            auto* p = static_cast<uint16_t*>(data);
            std::uniform_int_distribution<uint16_t> dist;
            for (size_t i = 0; i < count; ++i) p[i] = dist(g_rng);
            break;
        }
        case ACL_UINT32: {
            auto* p = static_cast<uint32_t*>(data);
            std::uniform_int_distribution<uint32_t> dist;
            for (size_t i = 0; i < count; ++i) p[i] = dist(g_rng);
            break;
        }
        case ACL_UINT64: {
            auto* p = static_cast<uint64_t*>(data);
            std::uniform_int_distribution<uint64_t> dist;
            for (size_t i = 0; i < count; ++i) p[i] = dist(g_rng);
            break;
        }
        case ACL_BOOL: {
            auto* p = static_cast<bool*>(data);
            std::bernoulli_distribution dist(0.5);
            for (size_t i = 0; i < count; ++i) p[i] = dist(g_rng);
            break;
        }
        // для комплексных и экзотических – заполняем float'ами пока
        case ACL_COMPLEX64: {
            float* p = static_cast<float*>(data);
            std::normal_distribution<float> dist(0.0f, 1.0f);
            for (size_t i = 0; i < count * 2; ++i) p[i] = dist(g_rng);
            break;
        }
        case ACL_COMPLEX128: {
            double* p = static_cast<double*>(data);
            std::normal_distribution<double> dist(0.0, 1.0);
            for (size_t i = 0; i < count * 2; ++i) p[i] = dist(g_rng);
            break;
        }
        default:
            // неизвестный тип – заполняем нулями
            memset(data, 0, count * aclDataTypeBytes(dtype));
    }
}


ACL_FUNC_VISIBILITY aclError aclopCompileAndExecute(const char *opType,
    int numInputs, const aclTensorDesc *const inputDesc[], const aclDataBuffer *const inputs[],
    int numOutputs, const aclTensorDesc *const outputDesc[], aclDataBuffer *const outputs[],
    const aclopAttr *attr, aclopEngineType engineType, aclopCompileType compileFlag,
    const char *opPath, aclrtStream stream) {

    std::ostringstream log;
    log << "[aclopCompileAndExecute] opType=" << (opType ? opType : "(null)")
        << " opPath=" << (opPath ? opPath : "(null)")
        << "\n    numInputs=" << numInputs << " numOutputs=" << numOutputs << '\n'
        << formatTensorList("input", inputDesc, numInputs)
        << formatTensorList("output", outputDesc, numOutputs)
        << "    stream=" << stream;
    log_output(log);

    // --- эмуляция операций ---
    if (opType && strstr(opType, "StatelessRandomNormalV2")) {
        for (int i = 0; i < numOutputs; ++i) {
            if (outputs[i] && outputDesc[i] && outputs[i]->data) {
                size_t count = calc_num_elements(outputDesc[i], outputs[i]->size);
                aclDataType dt = aclGetTensorDescType(outputDesc[i]);
                fill_random_normal(dt, outputs[i]->data, count);
            }
        }
    }

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
