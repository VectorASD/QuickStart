#include "common.h"     // log_output, ...

#include "op_profiler.h" // record_op_timing

#include "not_acl.cpp"  // aclGetTensorDescDimV2, aclGetTensorDescNumDims, aclGetTensorDescType
#include "helpers.cpp"
#include <cstring>      // size_t, strcpy
#include <string>       // std::string

#include <ATen/core/Generator.h>

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

constexpr int ACL_COMPILE_OPT_COUNT = static_cast<int>(ACL_OP_DEBUG_OPTION) + 1;
static std::string compileOptValues[ACL_COMPILE_OPT_COUNT];

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
 /* std::ostringstream log;
    log << "[aclSetCompileopt] opt=" << aclCompileOptToString(opt)
        << " (" << static_cast<int>(opt) << ")"
        << " value='" << (value ? value : "(null)") << '\'';
    log_output(log); */

    compileOptValues[opt] = (value != nullptr) ? value : std::string{};

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY size_t aclGetCompileoptSize(aclCompileOpt opt) {
    std::ostringstream log;
    log << "[aclGetCompileoptSize] opt=" << aclCompileOptToString(opt)
        << " (" << static_cast<int>(opt) << ")";
    log_output(log);

    return compileOptValues[opt].size() + 1; // +1 для '\0'
}

ACL_FUNC_VISIBILITY aclError aclGetCompileopt(aclCompileOpt opt, char *value, size_t length) {    
    std::ostringstream log;
    log << "[aclGetCompileopt] opt=" << aclCompileOptToString(opt)
        << " (" << static_cast<int>(opt) << ")"
        << " length=" << length
        << " value='" << compileOptValues[opt] << '\'';
    log_output(log);

    if (!value || length == 0)
        return ACL_ERROR_INVALID_PARAM;

    auto val = compileOptValues[opt];
    if (val.size() + 1 > length)
        return ACL_ERROR_INVALID_PARAM; // буфер слишком мал, использовался явно не aclGetCompileoptSize
    strcpy(value, val.c_str());
    return ACL_SUCCESS;
}

static std::string logCompileOpts() {
    std::ostringstream log;
    bool first = true;
    for (int i = 0; i < ACL_COMPILE_OPT_COUNT; ++i)
        if (!compileOptValues[i].empty()) {
            if (first)
                first = false;
            else
                log << ", ";
            log << aclCompileOptToString(static_cast<aclCompileOpt>(i))
                << "='" << compileOptValues[i] << '\'';
        }
    return log.str();
}

static bool _init = (compileOptValues[ACL_OP_JIT_COMPILE] = "disable", true);
// Иначе комплексные числа просто не заведутся


// aclopCompileAndExecute функционал

typedef enum {
    ACL_OP_COMPILE_DEFAULT = 0
} aclopCompileType;

typedef enum {
    ACL_ENGINE_SYS = 0
} aclopEngineType;


// Конструкторы тензоров

REGISTER_OP(StatelessRandomNormalV2, {
    // Входы:  shape (int64), seed (uint64), counter (uint64[2]), alg (int32)
    // Выходы: result (float/double/half/…)
    at::Tensor shape_tensor, out;
    ASSERT(numInputs == 4 && numOutputs == 1)                 // ровно 4 входа, 1 выход
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)   // выходной буфер существует и не пуст
    ASSERT(inputs[0] && inputDesc[0])                         // shape: может быть пустым (data = nullptr для скаляра)
    ASSERT(inputs[1] && inputDesc[1] && inputs[1]->data)      // seed: uint64, не пуст
    ASSERT(inputs[2] && inputDesc[2] && inputs[2]->data)      // counter: uint64[2], не пуст

    TRY(toAtenTensor(inputDesc[0], inputs[0], shape_tensor));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    const uint64_t* seed_ptr   = static_cast<const uint64_t*>(inputs[1]->data);
    const uint64_t* counter_ptr = static_cast<const uint64_t*>(inputs[2]->data);
    uint64_t effective_seed = *seed_ptr + counter_ptr[1];     // offset = counter[1]

    auto gen = at::detail::getDefaultCPUGenerator();
    gen.set_current_seed(effective_seed);

    at::IntArrayRef sizes(shape_tensor.data_ptr<int64_t>(), shape_tensor.numel());
    at::randn_out(out, sizes, gen);
    return H_OK;
});

REGISTER_OP(StatelessRandomUniformV2, {
    // Входы:  shape (int64), seed (uint64), counter (uint64[2]), alg (int32)
    // Выходы: result (float/double/half/…)
    at::Tensor shape_tensor, out;
    ASSERT(numInputs == 4 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(inputs[0] && inputDesc[0])                         // shape: может быть пустым (data = nullptr для скаляра)
    ASSERT(inputs[1] && inputDesc[1] && inputs[1]->data)      // seed: uint64, не пуст
    ASSERT(inputs[2] && inputDesc[2] && inputs[2]->data)      // counter: uint64[2], не пуст

    TRY(toAtenTensor(inputDesc[0], inputs[0], shape_tensor));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    const uint64_t* seed_ptr   = static_cast<const uint64_t*>(inputs[1]->data);
    const uint64_t* counter_ptr = static_cast<const uint64_t*>(inputs[2]->data);
    uint64_t effective_seed = *seed_ptr + counter_ptr[1];     // offset = counter[1]

    auto gen = at::detail::getDefaultCPUGenerator();
    gen.set_current_seed(effective_seed);

    at::IntArrayRef sizes(shape_tensor.data_ptr<int64_t>(), shape_tensor.numel());
    at::rand_out(out, sizes, gen);
    return H_OK;
});

REGISTER_OP(StatelessRandperm, {
    // Входы:  n (int64), seed (int64), offset (int64)
    // Выходы: перестановка (int64)
    at::Tensor n_tensor, out;
    ASSERT(numInputs == 3 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(inputs[0] && inputDesc[0])                         // n: int64, может быть пустым (data = nullptr для n=0)
    ASSERT(inputs[1] && inputDesc[1] && inputs[1]->data)      // seed: int64, не пуст
    ASSERT(inputs[2] && inputDesc[2] && inputs[2]->data)      // offset: int64, не пуст

    TRY(toAtenTensor(inputDesc[0], inputs[0], n_tensor));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    int64_t n = n_tensor.item<int64_t>();
    const int64_t* seed_ptr = static_cast<const int64_t*>(inputs[1]->data);
    const int64_t* offset_ptr = static_cast<const int64_t*>(inputs[2]->data);

    auto gen = at::detail::getDefaultCPUGenerator();
    gen.set_current_seed(*seed_ptr + *offset_ptr);

    at::randperm_out(out, n, gen);
    return H_OK;
});


REGISTER_OP(ZerosLike, {
    // Входы:  образец (форма и dtype)
    // Выходы: тензор из нулей той же формы и типа
    at::Tensor a, out;
    ASSERT(numInputs == 1 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)      // образец: любой dtype

    TRY(toAtenTensor(inputDesc[0], inputs[0], a));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    at::zeros_like_out(out, a);
    return H_OK;
});


REGISTER_OP(Eye, {
    // Входы:  отсутствуют (размеры и dtype определяются выходным дескриптором)
    // Выходы: единичная матрица [n, m]
    at::Tensor out;
    ASSERT(numInputs == 0 && numOutputs == 1)   // без входов, один выход
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)  // выходной тензор готов принять данные

    size_t ndim = aclGetTensorDescNumDims(outputDesc[0], false);
    ASSERT(ndim == 2)                           // выход – двумерная матрица

    int64_t n = 0, m = 0;
    ASSERT(aclGetTensorDescDimV2(outputDesc[0], 0, &n, false) == ACL_SUCCESS)  // число строк (n)
    ASSERT(aclGetTensorDescDimV2(outputDesc[0], 1, &m, false) == ACL_SUCCESS)  // число столбцов (m)

    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    at::eye_out(out, n, m);
    return H_OK;
});


REGISTER_OP(Fill, {
    // Входы:  тензор-образец (размеры задают выход), скалярное значение
    // Выходы: тензор, заполненный заданным значением
    at::Tensor dummy, value, out;
    ASSERT(numInputs == 2 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(inputs[0] && inputDesc[0])
    ASSERT(inputs[1] && inputDesc[1] && inputs[1]->data)

    TRY(toAtenTensor(inputDesc[0], inputs[0], dummy));
    TRY(toAtenTensor(inputDesc[1], inputs[1], value));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    at::Scalar fill_scalar = value.item();
    out.fill_(fill_scalar);
    return H_OK;
});


REGISTER_OP(OnesLike, {
    // Входы:  образец (форма и dtype)
    // Выходы: тензор из единиц той же формы и типа
    at::Tensor a, out;
    ASSERT(numInputs == 1 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)      // образец: любой dtype

    TRY(toAtenTensor(inputDesc[0], inputs[0], a));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    at::ones_like_out(out, a);
    return H_OK;
});


// Арифметика

REGISTER_OP(Mul, {
    // Входы:  a, b (совместимые типы)
    // Выходы: произведение a * b (с broadcasting)
    at::Tensor a, b, out;
    ASSERT(numInputs == 2 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)      // a
    ASSERT(inputs[1] && inputDesc[1] && inputs[1]->data)      // b

    TRY(toAtenTensor(inputDesc[0], inputs[0], a));
    TRY(toAtenTensor(inputDesc[1], inputs[1], b));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    at::mul_out(out, a, b);
    return H_OK;
});

REGISTER_OP(Add, {
    // Входы:  a, b
    // Выходы: сумма a + b (с broadcasting)
    at::Tensor a, b, out;
    ASSERT(numInputs == 2 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)      // a
    ASSERT(inputs[1] && inputDesc[1] && inputs[1]->data)      // b

    TRY(toAtenTensor(inputDesc[0], inputs[0], a));
    TRY(toAtenTensor(inputDesc[1], inputs[1], b));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    at::add_out(out, a, b);
    return H_OK;
});

REGISTER_OP(Sub, {
    // Входы:  a, b
    // Выходы: разность a - b (с broadcasting)
    at::Tensor a, b, out;
    ASSERT(numInputs == 2 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)      // a
    ASSERT(inputs[1] && inputDesc[1] && inputs[1]->data)      // b

    TRY(toAtenTensor(inputDesc[0], inputs[0], a));
    TRY(toAtenTensor(inputDesc[1], inputs[1], b));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    at::sub_out(out, a, b);
    return H_OK;
});

REGISTER_OP(RealDiv, {
    // Входы:  a, b
    // Выходы: частное a / b (с broadcasting)
    at::Tensor a, b, out;
    ASSERT(numInputs == 2 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)      // a
    ASSERT(inputs[1] && inputDesc[1] && inputs[1]->data)      // b

    TRY(toAtenTensor(inputDesc[0], inputs[0], a));
    TRY(toAtenTensor(inputDesc[1], inputs[1], b));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    at::div_out(out, a, b);
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


// Унарные операции

REGISTER_OP(Abs, {
    // Вход:  a (float/double/int/half/bf16) – тензор любого числового типа
    // Выход: out (тот же тип) – модуль элементов входного тензора
    at::Tensor a, out;
    ASSERT(numInputs == 1 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)      // a: тензор

    TRY(toAtenTensor(inputDesc[0], inputs[0], a));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    at::abs_out(out, a);
    return H_OK;
});

REGISTER_OP(AsStrided, {
    // Входы:  self (тензор), shape (int64[]), stride (int64[]), storage_offset (int64 скаляр)
    // Выходы: result (тензор с новыми shape и stride)
    at::Tensor self, result;
    ASSERT(numInputs == 4 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)   // result: тензор с заданной формой
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)      // self: входной тензор
    ASSERT(inputs[1] && inputDesc[1] && inputs[1]->data)      // shape: int64[]
    ASSERT(inputs[2] && inputDesc[2] && inputs[2]->data)      // stride: int64[]
    ASSERT(inputs[3] && inputDesc[3] && inputs[3]->data)      // storage_offset: int64 скаляр

    TRY(toAtenTensor(inputDesc[0], inputs[0], self));
    TRY(toAtenTensor(outputDesc[0], outputs[0], result));

    at::Tensor shape_tensor, stride_tensor, offset_tensor;
    TRY(toAtenTensor(inputDesc[1], inputs[1], shape_tensor));
    TRY(toAtenTensor(inputDesc[2], inputs[2], stride_tensor));
    TRY(toAtenTensor(inputDesc[3], inputs[3], offset_tensor));

    int64_t storage_offset = offset_tensor.item<int64_t>();
    auto shape_ptr = shape_tensor.data_ptr<int64_t>();
    auto stride_ptr = stride_tensor.data_ptr<int64_t>();
    int64_t ndim = shape_tensor.numel();

    at::as_strided_copy_out(result, self, at::IntArrayRef(shape_ptr, ndim),
                          at::IntArrayRef(stride_ptr, ndim), storage_offset);
    return H_OK;
});

REGISTER_OP(Cast, {
    at::Tensor inp, out;
    ASSERT(numInputs == 1 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0])  // буфер может быть nullptr для пустого тензора
    ASSERT(inputs[0] && inputDesc[0])

    TRY(toAtenTensor(inputDesc[0], inputs[0], inp));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    if (out.numel() > 0)
        out.copy_(inp.to(out.options()));  // нет to_out
    return H_OK;
});

REGISTER_OP(Ceil, {
    // Вход:  a (float/double/half/bf16) – тензор с плавающей точкой
    // Выход: out (тот же тип) – округление вверх до целого
    at::Tensor a, out;
    ASSERT(numInputs == 1 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)      // a: тензор с плавающей точкой

    TRY(toAtenTensor(inputDesc[0], inputs[0], a));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    at::ceil_out(out, a);
    return H_OK;
});


REGISTER_OP(LeftShift, {
    // Входы: self (целочисленный тензор), other (целочисленный тензор/скаляр)
    // Выходы: result (тот же тип)
    at::Tensor self, other, out;
    ASSERT(numInputs == 2 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)
    ASSERT(inputs[1] && inputDesc[1] && inputs[1]->data)

    TRY(toAtenTensor(inputDesc[0], inputs[0], self));
    TRY(toAtenTensor(inputDesc[1], inputs[1], other));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    ASSERT_CODE(self.scalar_type() == other.scalar_type() &&
                at::isIntegralType(self.scalar_type(), /*includeBool=*/false),
                H_UNIMPLEMENTED);

    at::bitwise_left_shift_out(out, self, other);
    return H_OK;
});

REGISTER_OP(RightShift, {
    // Входы: self (целочисленный тензор), other (целочисленный тензор/скаляр)
    // Выходы: result (тот же тип)
    at::Tensor self, other, out;
    ASSERT(numInputs == 2 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)
    ASSERT(inputs[1] && inputDesc[1] && inputs[1]->data)

    TRY(toAtenTensor(inputDesc[0], inputs[0], self));
    TRY(toAtenTensor(inputDesc[1], inputs[1], other));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    ASSERT_CODE(self.scalar_type() == other.scalar_type() &&
                at::isIntegralType(self.scalar_type(), /*includeBool=*/false),
                H_UNIMPLEMENTED);

    at::bitwise_right_shift_out(out, self, other);
    return H_OK;
});


REGISTER_OP(Invert, {
    // Входы: self (целочисленный тензор)
    // Выходы: result (тот же тип)
    at::Tensor self, out;
    ASSERT(numInputs == 1 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)

    TRY(toAtenTensor(inputDesc[0], inputs[0], self));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    ASSERT_CODE(at::isIntegralType(self.scalar_type(), /*includeBool=*/false),
                H_UNIMPLEMENTED);

    at::bitwise_not_out(out, self);
    return H_OK;
});

REGISTER_OP(LogicalNot, {
    // Входы: self (bool тензор)
    // Выходы: result (bool тензор)
    at::Tensor self, out;
    ASSERT(numInputs == 1 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)

    TRY(toAtenTensor(inputDesc[0], inputs[0], self));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    ASSERT_CODE(self.scalar_type() == at::kBool &&
                out.scalar_type() == at::kBool,
                H_UNIMPLEMENTED);

    at::logical_not_out(out, self);
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


REGISTER_OP(Cos, {
    // Входы: self (float/double/half/bf16 тензор)
    // Выходы: result (тот же тип)
    at::Tensor self, out;
    ASSERT(numInputs == 1 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)

    TRY(toAtenTensor(inputDesc[0], inputs[0], self));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    ASSERT_CODE(at::isFloatingType(self.scalar_type()),
                H_UNIMPLEMENTED);

    at::cos_out(out, self);
    return H_OK;
});

REGISTER_OP(Cosh, {
    // Входы: self (float/double/half/bf16 тензор)
    // Выходы: result (тот же тип)
    at::Tensor self, out;
    ASSERT(numInputs == 1 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)

    TRY(toAtenTensor(inputDesc[0], inputs[0], self));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    ASSERT_CODE(at::isFloatingType(self.scalar_type()),
                H_UNIMPLEMENTED);

    at::cosh_out(out, self);
    return H_OK;
});

REGISTER_OP(Acos, {
    // Вход:  self (float/double/half/bf16) – тензор, к которому применяется арккосинус
    // Выход: result (тот же тип) – тензор с вычисленным acos(self)
    at::Tensor self, result;
    ASSERT(numInputs == 1 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)   // result: тензор плавающего типа
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)      // self: тензор плавающего типа

    TRY(toAtenTensor(inputDesc[0], inputs[0], self));
    TRY(toAtenTensor(outputDesc[0], outputs[0], result));

    at::acos_out(result, self);
    return H_OK;
});

REGISTER_OP(Acosh, {
    // Вход:  self (float/double/half/bf16) – тензор, к которому применяется гиперболический арккосинус
    // Выход: result (тот же тип) – тензор с вычисленным acosh(self)
    at::Tensor self, result;
    ASSERT(numInputs == 1 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)   // result: тензор плавающего типа
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)      // self: тензор плавающего типа

    TRY(toAtenTensor(inputDesc[0], inputs[0], self));
    TRY(toAtenTensor(outputDesc[0], outputs[0], result));

    at::acosh_out(result, self);
    return H_OK;
});


REGISTER_OP(Exp, {
    // Входы: self (float/double/half/bf16 тензор)
    // Выходы: result (тот же тип)
    at::Tensor self, out;
    ASSERT(numInputs == 1 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)

    TRY(toAtenTensor(inputDesc[0], inputs[0], self));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    ASSERT_CODE(at::isFloatingType(self.scalar_type()),
                H_UNIMPLEMENTED);

    at::exp_out(out, self);
    return H_OK;
});

REGISTER_OP(Expm1, {
    // Входы: self (float/double/half/bf16 тензор)
    // Выходы: result (тот же тип)
    at::Tensor self, out;
    ASSERT(numInputs == 1 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)

    TRY(toAtenTensor(inputDesc[0], inputs[0], self));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    ASSERT_CODE(at::isFloatingType(self.scalar_type()),
                H_UNIMPLEMENTED);

    at::expm1_out(out, self);
    return H_OK;
});


REGISTER_OP(Pow, {
    // Входы:
    //   - self (float/double/half/bf16 тензор)
    //   - exp  (float/double/half/bf16 тензор или скаляр, переданный как тензор размера [1])
    // Выходы: result (тип определяется через at::result_type(self, exp))
    at::Tensor self, exp, out;
    ASSERT(numInputs == 2 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)
    ASSERT(inputs[1] && inputDesc[1] && inputs[1]->data)

    TRY(toAtenTensor(inputDesc[0], inputs[0], self));
    TRY(toAtenTensor(inputDesc[1], inputs[1], exp));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    ASSERT_CODE(at::isFloatingType(self.scalar_type()) &&
                at::isFloatingType(exp.scalar_type()),
                H_UNIMPLEMENTED);

    at::pow_out(out, self, exp);
    return H_OK;
});


REGISTER_OP(FastGelu, {
    // Входы: self (float/double/half/bf16 тензор)
    // Выходы: result (тот же тип)
    at::Tensor self, out;
    ASSERT(numInputs == 1 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)

    TRY(toAtenTensor(inputDesc[0], inputs[0], self));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    ASSERT_CODE(at::isFloatingType(self.scalar_type()),
                H_UNIMPLEMENTED);

    at::gelu_out(out, self, "tanh");
    return H_OK;
});

REGISTER_OP(Gelu, {
    // Входы: self (float/double/half/bf16 тензор)
    // Атрибуты (опционально): approximate (int64) – 0 для erf, 1 для tanh
    // Выходы: result (тот же тип)
    at::Tensor self, out;
    ASSERT(numInputs == 1 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)

    TRY(toAtenTensor(inputDesc[0], inputs[0], self));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    ASSERT_CODE(at::isFloatingType(self.scalar_type()),
                H_UNIMPLEMENTED);

    // Определяем режим: по умолчанию erf, если атрибут approximate=1 → tanh
    bool use_tanh = false;
    if (attr) {
        auto it = attr->ints.find("approximate");
        if (it != attr->ints.end() && it->second == 1)
            use_tanh = true;
    }
    at::gelu_out(out, self, use_tanh ? "tanh" : "none");
    return H_OK;
});

REGISTER_OP(FastGeluGrad, {
    // Входы: grad (float/double/half/bf16 тензор), self (тот же тип)
    // Выходы: grad_input (тот же тип)
    at::Tensor grad, self, grad_input;
    ASSERT(numInputs == 2 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)
    ASSERT(inputs[1] && inputDesc[1] && inputs[1]->data)

    TRY(toAtenTensor(inputDesc[0], inputs[0], grad));
    TRY(toAtenTensor(inputDesc[1], inputs[1], self));
    TRY(toAtenTensor(outputDesc[0], outputs[0], grad_input));

    ASSERT_CODE(at::isFloatingType(grad.scalar_type()) &&
                grad.scalar_type() == self.scalar_type(),
                H_UNIMPLEMENTED);

    grad_input.copy_(at::gelu_backward(grad, self, "tanh"));
    return H_OK;
});

REGISTER_OP(GeluGrad, {
    // Входы: grad (float/double/half/bf16 тензор), self (тот же тип)
    // Выходы: grad_input (тот же тип)
    at::Tensor grad, self, grad_input;
    ASSERT(numInputs == 2 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)
    ASSERT(inputs[1] && inputDesc[1] && inputs[1]->data)

    TRY(toAtenTensor(inputDesc[0], inputs[0], grad));
    TRY(toAtenTensor(inputDesc[1], inputs[1], self));
    TRY(toAtenTensor(outputDesc[0], outputs[0], grad_input));

    ASSERT_CODE(at::isFloatingType(grad.scalar_type()) &&
                grad.scalar_type() == self.scalar_type(),
                H_UNIMPLEMENTED);

    grad_input.copy_(at::gelu_backward(grad, self));
    return H_OK;
});


// Редукция

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


// Маски

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


// Остальное

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
    // Входы:  N тензоров одного типа
    // Выходы: конкатенация вдоль заданной оси
    int N;
    int64_t concat_dim;
    std::vector<at::Tensor> tensors;
    at::Tensor out_tensor;

    ASSERT(numInputs >= 1 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(attr)
    ASSERT(try_get_attr<int>(attr, "N", N) && N == numInputs)    // число тензоров
    ASSERT(try_get_attr<int64_t>(attr, "concat_dim", concat_dim))

    TRY(toAtenTensors(N, inputDesc, inputs, tensors));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out_tensor));

    at::cat_out(out_tensor, tensors, concat_dim);
    return H_OK;
});

REGISTER_OP(Pack, {
    // Входы:  N тензоров одинакового размера
    // Выходы: упаковка в новый тензор вдоль новой оси
    int N;
    int64_t axis;
    std::vector<at::Tensor> tensors;
    at::Tensor out_tensor;

    ASSERT(numInputs >= 1 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(attr)
    ASSERT(try_get_attr<int>(attr, "N", N) && N == numInputs)
    ASSERT(try_get_attr<int64_t>(attr, "axis", axis))

    TRY(toAtenTensors(N, inputDesc, inputs, tensors));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out_tensor));

    at::stack_out(out_tensor, tensors, axis);
    return H_OK;
});


REGISTER_OP(Sort, {
    // Вход:  self (любой тип)
    // Выход: values (тот же тип), indices (int64)
    at::Tensor self, values_out, indices_out;
    ASSERT(numInputs == 1 && numOutputs == 2)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)   // values
    ASSERT(outputs[1] && outputDesc[1] && outputs[1]->data)   // indices
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)

    int64_t axis;
    bool descending;
    ASSERT(attr)
    ASSERT(try_get_attr<int64_t>(attr, "axis", axis))          // ось сортировки
    ASSERT(try_get_attr<bool>(attr, "descending", descending)) // направление

    TRY(toAtenTensor(inputDesc[0], inputs[0], self));
    TRY(toAtenTensor(outputDesc[0], outputs[0], values_out));
    TRY(toAtenTensor(outputDesc[1], outputs[1], indices_out));

    // terminate called after throwing an instance of 'c10::Error'
    //  what():  Expected out tensor to have dtype long int, but got int instead
    // Exception raised from resize_out at /pytorch/build/aten/src/ATen/RegisterCPU_0.cpp:1106 (most recent call first): ...

    if (indices_out.scalar_type() != at::kLong) {
        auto [sorted_vals, sorted_idxs] = at::sort(self, axis, descending);
        values_out.copy_(sorted_vals);
        indices_out.copy_(sorted_idxs);
    } else
        at::sort_out(values_out, indices_out, self, axis, descending);
    return H_OK;
});

REGISTER_OP(SortV2, {
    // Вход:  self
    // Выход: values (тот же тип)
    at::Tensor self, values_out;
    ASSERT(numInputs == 1 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)   // values
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)

    int64_t axis;
    bool descending;
    ASSERT(attr)
    ASSERT(try_get_attr<int64_t>(attr, "axis", axis))
    ASSERT(try_get_attr<bool>(attr, "descending", descending))

    TRY(toAtenTensor(inputDesc[0], inputs[0], self));
    TRY(toAtenTensor(outputDesc[0], outputs[0], values_out));

    at::Tensor dummy_indices = at::empty_like(values_out, values_out.options().dtype(at::kLong));
    at::sort_out(values_out, dummy_indices, self, axis, descending);
    return H_OK;
});


REGISTER_OP(OneHot, {
    // Входы: self (float/int, будет приведён к Long), depth (int), on_value (скаляр), off_value (скаляр)
    // Выход: one‑hot тензор (тип совпадает с self? в тестах – Long)
    at::Tensor self, depth_tensor, out;
    ASSERT(numInputs == 4 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0])                        // выход может быть не пустым
    ASSERT(inputs[0] && inputDesc[0])                          // self: может быть пустым, data может быть nullptr
    ASSERT(inputs[1] && inputDesc[1] && inputs[1]->data)      // depth: int, всегда есть данные
    ASSERT(inputs[2] && inputDesc[2] && inputs[2]->data)      // on_value: скаляр, всегда есть данные
    ASSERT(inputs[3] && inputDesc[3] && inputs[3]->data)      // off_value: скаляр, всегда есть данные

    TRY(toAtenTensor(inputDesc[0], inputs[0], self));          // toAtenTensor обрабатывает nullptr для пустых тензоров
    TRY(toAtenTensor(inputDesc[1], inputs[1], depth_tensor));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    int64_t depth = depth_tensor.item<int64_t>();
    at::Tensor result = at::one_hot(self.to(at::kLong), depth).to(out.options());
    if (out.numel() > 0)
        out.copy_(result);  // нет one_hot_out
    return H_OK;
});

REGISTER_OP(OneHotD, {
    // Входы: self_copy (int), on_tmp (float [1]), off_tmp (float [1])
    // Выход: one‑hot тензор (float)
    at::Tensor self_int, out;
    ASSERT(numInputs == 3 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0])                        // выход может быть не пустым
    ASSERT(inputs[0] && inputDesc[0])                          // self_copy: может быть пустым, data м.б. nullptr
    ASSERT(inputs[1] && inputDesc[1] && inputs[1]->data)      // on_tmp: скаляр [1], данные есть всегда
    ASSERT(inputs[2] && inputDesc[2] && inputs[2]->data)      // off_tmp: скаляр [1], данные есть всегда

    int64_t axis, depth;
    ASSERT(attr)
    ASSERT(try_get_attr<int64_t>(attr, "axis", axis))
    ASSERT(try_get_attr<int64_t>(attr, "depth", depth))

    TRY(toAtenTensor(inputDesc[0], inputs[0], self_int));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    at::Tensor result = at::one_hot(self_int, depth).to(out.options());
    if (out.numel() > 0)
        out.copy_(result);  // нет one_hot_out
    return H_OK;
});


REGISTER_OP(Slice, {
    // Входы: self (любой тензор), offsets (int64 тензор), size (int64 тензор)
    // Выходы: result (тот же тип, форма = size)
    at::Tensor self, offsets, sizes, out;
    ASSERT(numInputs == 3 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)
    ASSERT(inputs[1] && inputDesc[1] && inputs[1]->data)
    ASSERT(inputs[2] && inputDesc[2] && inputs[2]->data)

    TRY(toAtenTensor(inputDesc[0], inputs[0], self));
    TRY(toAtenTensor(inputDesc[1], inputs[1], offsets));
    TRY(toAtenTensor(inputDesc[2], inputs[2], sizes));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    ASSERT_CODE(offsets.scalar_type() == at::kLong &&
                sizes.scalar_type() == at::kLong &&
                offsets.dim() == 1 && sizes.dim() == 1 &&
                offsets.size(0) == self.dim(),
                H_UNASSERTED);

    // Эмуляция через последовательное narrow
    at::Tensor sliced = self;
    auto offsets_acc = offsets.accessor<int64_t, 1>();
    auto sizes_acc   = sizes.accessor<int64_t, 1>();
    for (int64_t d = 0; d < self.dim(); ++d) {
        sliced = sliced.narrow(d, offsets_acc[d], sizes_acc[d]);
    }
    out.copy_(sliced);
    return H_OK;
});

REGISTER_OP(StridedSlice, {
    // Входы: self (любой тензор), begin (int64 вектор), end (int64 вектор), strides (int64 вектор)
    // Атрибуты: begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask (int64)
    // Выходы: result (форма определяется согласно правилам StridedSlice)
    at::Tensor self, begin, end, strides, out;
    ASSERT(numInputs == 4 && numOutputs == 1)
    ASSERT(outputs[0] && outputDesc[0] && outputs[0]->data)
    ASSERT(inputs[0] && inputDesc[0] && inputs[0]->data)
    ASSERT(inputs[1] && inputDesc[1] && inputs[1]->data)
    ASSERT(inputs[2] && inputDesc[2] && inputs[2]->data)
    ASSERT(inputs[3] && inputDesc[3] && inputs[3]->data)
    ASSERT(attr)

    TRY(toAtenTensor(inputDesc[0], inputs[0], self));
    TRY(toAtenTensor(inputDesc[1], inputs[1], begin));
    TRY(toAtenTensor(inputDesc[2], inputs[2], end));
    TRY(toAtenTensor(inputDesc[3], inputs[3], strides));
    TRY(toAtenTensor(outputDesc[0], outputs[0], out));

    ASSERT_CODE(begin.scalar_type() == at::kLong &&
                end.scalar_type() == at::kLong &&
                strides.scalar_type() == at::kLong,
                H_UNASSERTED);

    int64_t begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask;
    ASSERT(try_get_attr<int64_t>(attr, "begin_mask", begin_mask));
    ASSERT(try_get_attr<int64_t>(attr, "end_mask", end_mask));
    ASSERT(try_get_attr<int64_t>(attr, "ellipsis_mask", ellipsis_mask));
    ASSERT(try_get_attr<int64_t>(attr, "new_axis_mask", new_axis_mask));
    ASSERT(try_get_attr<int64_t>(attr, "shrink_axis_mask", shrink_axis_mask));

    auto begin_acc = begin.accessor<int64_t, 1>();
    auto end_acc   = end.accessor<int64_t, 1>();
    auto strides_acc = strides.accessor<int64_t, 1>();

    int64_t num_specified = begin.size(0);          // число заданных осей в begin/end/strides
    int64_t num_new_axes = __builtin_popcountll(new_axis_mask);
    int64_t num_shrink_axes = __builtin_popcountll(shrink_axis_mask);
    // Итоговое количество измерений, которое должно получиться:
    // исходный ранг self + new_axes - shrink_axes.
    // При наличии эллипсиса недостающие оси заполняются из self, но здесь всё учтено.

    // Восстанавливаем полные списки для каждой оси (после эллипсиса)
    std::vector<int64_t> full_begin, full_end, full_strides;
    std::vector<bool> is_new_axis, is_shrink_axis;

    int64_t spec_idx = 0;       // индекс в массивах begin/end/strides
    int64_t self_axis = 0;      // текущая ось self, на которую проецируется спецификация
    bool ellipsis_seen = false;
    int64_t self_rank = self.dim();
    int64_t total_axes_after_ellipsis = self_rank + num_new_axes - num_shrink_axes;

    // Парсим спецификацию, обрабатывая эллипсис
    for (int64_t i = 0; i < num_specified; ++i) {
        if (ellipsis_mask & (1ULL << i)) {
            // Вставляем эллипсис: добавляем все оставшиеся оси self
            while (self_axis < self_rank) {
                full_begin.push_back(0);
                full_end.push_back(self.size(self_axis));
                full_strides.push_back(1);
                is_new_axis.push_back(false);
                is_shrink_axis.push_back(false);
                ++self_axis;
            }
            ellipsis_seen = true;
            continue;
        }
        if (new_axis_mask & (1ULL << i)) {
            // new axis
            full_begin.push_back(0);
            full_end.push_back(1);
            full_strides.push_back(1);
            is_new_axis.push_back(true);
            is_shrink_axis.push_back(false);
            // не увеличиваем self_axis
        } else {
            // обычная или shrink ось
            bool shrink = (shrink_axis_mask & (1ULL << i));
            // Проверка выхода за границы self_rank
            if (self_axis >= self_rank) return H_UNASSERTED;
            int64_t dim_size = self.size(self_axis);
            int64_t b = (begin_mask & (1ULL << i)) ? 0 : begin_acc[i];
            int64_t e = (end_mask & (1ULL << i)) ? dim_size : end_acc[i];
            int64_t s = strides_acc[i];
            // Корректируем отрицательные индексы
            if (b < 0) b += dim_size;
            if (e < 0) e += dim_size;
            // При отрицательном stride меняем b и e местами, чтобы выполнялось b < e для slice
            // PyTorch slice поддерживает отрицательные stride, корректировка не обязательна,
            // но для единообразия и безопасности оставим.
            if (s < 0) {
                if (b == dim_size) b = dim_size - 1;
                if (e == -1) e = 0;
                // дополнительно ничего не меняем, slice обработает
            }
            full_begin.push_back(b);
            full_end.push_back(e);
            full_strides.push_back(s);
            is_new_axis.push_back(false);
            is_shrink_axis.push_back(shrink);
            ++self_axis;
        }
    }
    // Если эллипсис не встречался, но остались нераспределённые оси self
    if (!ellipsis_seen && self_axis < self_rank)
        while (self_axis < self_rank) {
            full_begin.push_back(0);
            full_end.push_back(self.size(self_axis));
            full_strides.push_back(1);
            is_new_axis.push_back(false);
            is_shrink_axis.push_back(false);
            ++self_axis;
        }

    // Теперь формируем список индексов для оператора []
    std::vector<at::indexing::TensorIndex> indices;
    int64_t res_axis = 0;  // ось в результирующем тензоре (после применения всех операций)
    int64_t full_len = full_begin.size();
    for (int64_t i = 0; i < full_len; ++i)
        if (is_new_axis[i])
            indices.push_back(at::indexing::None);
        else if (is_shrink_axis[i])
            // shrink: выбираем конкретный элемент (индекс из begin)
            // begin уже скорректирован с учётом отрицательных значений
            indices.push_back(full_begin[i]);
        else
            // обычный slice
            indices.push_back(at::indexing::Slice(full_begin[i], full_end[i], full_strides[i]));

    at::Tensor sliced;
    try {
        sliced = self.index(indices);
    } catch (const std::exception& e) {
        log_output(std::string("StridedSlice failed: ") + e.what(), true);
        return H_UNASSERTED;
    }

    // shrink_axis уже убрал размерность, new_axis добавил – должно совпасть с outputDesc
    out.copy_(sliced);
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
        << "\n    opts: " << logCompileOpts()
        << "\n    numInputs=" << numInputs << " numOutputs=" << numOutputs
    << formatTensorList("input", inputDesc, inputs, numInputs, PRINT_ALL)
    << formatTensorList("output", outputDesc, outputs, numOutputs, PRINT_DESC);
    try {

        auto t_start = std::chrono::steady_clock::now();

        OpHandler handler;
        exitCode code = H_UNKNOWN_OP;
        if (opType && OpRegistry::try_find(opType, handler))
            code = handler(numInputs, inputDesc, inputs, numOutputs, outputDesc, outputs, attr);

        auto t_end = std::chrono::steady_clock::now();
        double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();
        record_op_timing(opType, elapsed_us);

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
    } catch (const std::exception& e) {
        log << "\nError: !!! C++ exception in handler for " << opType << ": " << e.what();
        log_output(log, true);
        return ACL_ERROR_INTERNAL_ERROR;
    } catch (...) {
        log << "\nError: !!! Unknown C++ exception in handler for " << opType;
        log_output(log, true);
        return ACL_ERROR_INTERNAL_ERROR;
    }
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
