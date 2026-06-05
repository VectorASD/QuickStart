#include "not_acl.cpp"  // aclGetTensorDescDimV2, aclGetTensorDescNumDims, aclGetTensorDescType
#include <random>       // bernoulli_distribution, mt19937, normal_distribution, random_device, uniform_int_distribution
#include <algorithm>    // copy, min, sort

#include <ATen/ATen.h>

#ifndef HELPERS
#define HELPERS

static size_t calc_num_elements(const aclTensorDesc* desc, size_t bufferSize) {
    if (!desc) return 0;
    size_t numDims = aclGetTensorDescNumDims(desc, false);
    size_t numElements = 1;
    if (numDims == ACL_UNKNOWN_RANK) {
        // вычисляем по размеру буфера и типу
        aclDataType dt = aclGetTensorDescType(desc, false);
        size_t elemSize = aclDataTypeBytes(dt);
        if (elemSize > 0) numElements = bufferSize / elemSize;
        else numElements = 0;
    } else {
        for (size_t d = 0; d < numDims; ++d) {
            int64_t dimSize;
            if (aclGetTensorDescDimV2(desc, d, &dimSize, false) == ACL_SUCCESS)
                numElements *= dimSize;
        }
    }
    return numElements;
}

static std::vector<int64_t> compute_strides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size());
    if (!dims.empty()) {
        strides.back() = 1;
        for (int i = dims.size()-2; i >= 0; --i)
            strides[i] = strides[i+1] * dims[i];
    }
    return strides;
}

static const char* aclDataTypeToString(aclDataType dtype) {
    switch (dtype) {
        case ACL_FLOAT:           return "float32";
        case ACL_FLOAT16:         return "float16";
        case ACL_INT8:            return "int8";
        case ACL_INT32:           return "int32";
        case ACL_UINT8:           return "uint8";
        case ACL_INT16:           return "int16";
        case ACL_UINT16:          return "uint16";
        case ACL_UINT32:          return "uint32";
        case ACL_INT64:           return "int64";
        case ACL_UINT64:          return "uint64";
        case ACL_DOUBLE:          return "float64";
        case ACL_BOOL:            return "bool";
        case ACL_STRING:          return "string";
        case ACL_COMPLEX64:       return "complex64";
        case ACL_COMPLEX128:      return "complex128";
        case ACL_BF16:            return "bfloat16";
        case ACL_INT4:            return "int4";
        case ACL_UINT1:           return "uint1";
        case ACL_COMPLEX32:       return "complex32";
        case ACL_HIFLOAT8:        return "hifloat8";
        case ACL_FLOAT8_E5M2:     return "float8_e5m2";
        case ACL_FLOAT8_E4M3FN:   return "float8_e4m3fn";
        case ACL_FLOAT8_E8M0:     return "float8_e8m0";
        case ACL_FLOAT6_E3M2:     return "float6_e3m2";
        case ACL_FLOAT6_E2M3:     return "float6_e2m3";
        case ACL_FLOAT4_E2M1:     return "float4_e2m1";
        case ACL_FLOAT4_E1M2:     return "float4_e1m2";
        case ACL_HIFLOAT4:        return "hifloat4";
        default:                  return "unknown";
    }
}

static const char* aclFormatToString(aclFormat fmt) {
    switch (fmt) {
        case ACL_FORMAT_UNDEFINED:    return "UNDEFINED";
        case ACL_FORMAT_NCHW:         return "NCHW";
        case ACL_FORMAT_NHWC:         return "NHWC";
        case ACL_FORMAT_ND:           return "ND";
        case ACL_FORMAT_NC1HWC0:      return "NC1HWC0";
        case ACL_FORMAT_FRACTAL_Z:    return "FRACTAL_Z";
        case ACL_FORMAT_NC1HWC0_C04:  return "NC1HWC0_C04";
        case ACL_FORMAT_HWCN:         return "HWCN";
        case ACL_FORMAT_NDHWC:        return "NDHWC";
        case ACL_FORMAT_FRACTAL_NZ:   return "FRACTAL_NZ";
        case ACL_FORMAT_NCDHW:        return "NCDHW";
        case ACL_FORMAT_NDC1HWC0:     return "NDC1HWC0";
        case ACL_FRACTAL_Z_3D:        return "FRACTAL_Z_3D";
        case ACL_FORMAT_NC:           return "NC";
        case ACL_FORMAT_NCL:          return "NCL";
        case ACL_FORMAT_FRACTAL_NZ_C0_16: return "FRACTAL_NZ_C0_16";
        case ACL_FORMAT_FRACTAL_NZ_C0_32: return "FRACTAL_NZ_C0_32";
        case ACL_FORMAT_FRACTAL_NZ_C0_2:  return "FRACTAL_NZ_C0_2";
        case ACL_FORMAT_FRACTAL_NZ_C0_4:  return "FRACTAL_NZ_C0_4";
        case ACL_FORMAT_FRACTAL_NZ_C0_8:  return "FRACTAL_NZ_C0_8";
        default:                      return "unknown";
    }
}

static void tensorDescToString(const aclTensorDesc* desc, std::ostringstream &oss) {
    if (!desc) {
        oss << "null";
        return;
    }
    oss << aclDataTypeToString(desc->dtype) << '[';
    if (desc->dims.empty()) {
        oss << "scalar";
    } else {
        for (size_t i = 0; i < desc->dims.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << desc->dims[i];
        }
    }
    oss << ']';
    oss << " fmt=" << aclFormatToString(desc->format);
    oss << " mem=" << aclMemTypeToString(desc->memType);
    if (!desc->name.empty()) oss << " name='" << desc->name << "'";
}
static std::string tensorDescToString(const aclTensorDesc* desc) {
    std::ostringstream oss;
    tensorDescToString(desc, oss);
    return oss.str();
}

// Преобразует один элемент тензора в строку по его типу
static std::string aclElementToString(aclDataType dtype, const void* elemPtr) {
    if (!elemPtr)
        return "?";
    switch (dtype) {
        case ACL_FLOAT: {
            float val = *static_cast<const float*>(elemPtr);
            std::ostringstream oss;
            if (std::abs(val) >= 1e-4f && std::abs(val) < 1e4f)
                oss << std::fixed << std::setprecision(4) << val;
            else
                oss << std::scientific << std::setprecision(4) << val;
            return oss.str();
        }
        case ACL_DOUBLE: {
            double val = *static_cast<const double*>(elemPtr);
            std::ostringstream oss;
            if (std::abs(val) >= 1e-4 && std::abs(val) < 1e4)
                oss << std::fixed << std::setprecision(4) << val;
            else
                oss << std::scientific << std::setprecision(4) << val;
            return oss.str();
        }
        case ACL_FLOAT16:
        case ACL_BF16: {
            uint16_t v = *static_cast<const uint16_t*>(elemPtr);
            uint32_t bits = static_cast<uint32_t>(v) << 16;
            float f;
            std::memcpy(&f, &bits, sizeof(f));
            std::ostringstream oss;
            if (std::abs(f) >= 1e-4f && std::abs(f) < 1e4f)
                oss << std::fixed << std::setprecision(4) << f;
            else
                oss << std::scientific << std::setprecision(4) << f;
            return oss.str();
        }
        case ACL_INT8:
            return std::to_string(static_cast<int>(*static_cast<const int8_t*>(elemPtr)));
        case ACL_UINT8:
            return std::to_string(static_cast<unsigned>(*static_cast<const uint8_t*>(elemPtr)));
        case ACL_INT16:
            return std::to_string(*static_cast<const int16_t*>(elemPtr));
        case ACL_UINT16:
            return std::to_string(*static_cast<const uint16_t*>(elemPtr));
        case ACL_INT32:
            return std::to_string(*static_cast<const int32_t*>(elemPtr));
        case ACL_UINT32:
            return std::to_string(*static_cast<const uint32_t*>(elemPtr));
        case ACL_INT64:
            return std::to_string(*static_cast<const int64_t*>(elemPtr));
        case ACL_UINT64:
            return std::to_string(*static_cast<const uint64_t*>(elemPtr));
        case ACL_BOOL:
            return *static_cast<const bool*>(elemPtr) ? "true" : "false";
        default:
            return "?";
    }
}

static std::string tensorDataToString(const aclTensorDesc* desc, const aclDataBuffer* buf) {
    if (!desc || !buf || !buf->data || buf->size == 0)
        return "(no data)";

    size_t numElements = calc_num_elements(desc, buf->size);
    if (numElements == 0) return "(no elements)";

    const std::vector<int64_t>& dims = desc->dims;
    const int baseIndent = 8;
    const int edgeItems = 3;          // сколько элементов показывать с краёв при сокращении
    const size_t threshold = 1000;    // порог общего числа элементов для включения сокращения
    bool doTruncate = numElements > threshold;

    std::ostringstream oss;
    oss << std::string(baseIndent, ' ');
    if (dims.empty()) {
        oss << aclElementToString(desc->dtype, buf->data);
        return oss.str();
    }

    // Рекурсивная лямбда для форматирования
    std::function<void(std::ostringstream&, const void*, const std::vector<int64_t>&, int, int)> printRec;
    printRec = [&](std::ostringstream& oss, const void* data, const std::vector<int64_t>& curDims, int depth, int indent) {
        if (depth == curDims.size() - 1) {
            // Последнее измерение – строка чисел
            int64_t size = curDims[depth];
            oss << '[';
            if (doTruncate && size > 2 * edgeItems + 1) {
                for (int64_t i = 0; i < edgeItems; ++i) {
                    if (i > 0) oss << ", ";
                    oss << aclElementToString(desc->dtype, static_cast<const char*>(data) + i * aclDataTypeBytes(desc->dtype));
                }
                oss << ", ..., ";
                for (int64_t i = size - edgeItems; i < size; ++i) {
                    if (i > size - edgeItems) oss << ", ";
                    oss << aclElementToString(desc->dtype, static_cast<const char*>(data) + i * aclDataTypeBytes(desc->dtype));
                }
            } else {
                for (int64_t i = 0; i < size; ++i) {
                    if (i > 0) oss << ", ";
                    oss << aclElementToString(desc->dtype, static_cast<const char*>(data) + i * aclDataTypeBytes(desc->dtype));
                }
            }
            oss << ']';
        } else {
            // Промежуточное измерение – вывод вложенных тензоров
            indent++;
            auto pad = std::string(indent, ' ');
            int64_t size = curDims[depth];
            oss << '[';
            if (doTruncate && size > 2 * edgeItems + 1) {
                // Первые 3
                for (int64_t i = 0; i < edgeItems; ++i) {
                    if (i > 0) oss << ",\n" << pad;
                    size_t stride = aclDataTypeBytes(desc->dtype);
                    for (int d = depth + 1; d < curDims.size(); ++d) stride *= curDims[d];
                    const void* subData = static_cast<const char*>(data) + i * stride;
                    printRec(oss, subData, curDims, depth + 1, indent);
                }
                oss << ",\n" << std::string(indent + 1, ' ') << "...";
                // Последние 3
                for (int64_t i = size - edgeItems; i < size; ++i) {
                    oss << ",\n" << pad;
                    size_t stride = aclDataTypeBytes(desc->dtype);
                    for (int d = depth + 1; d < curDims.size(); ++d) stride *= curDims[d];
                    const void* subData = static_cast<const char*>(data) + i * stride;
                    printRec(oss, subData, curDims, depth + 1, indent);
                }
            } else {
                for (int64_t i = 0; i < size; ++i) {
                    if (i > 0) oss << ",\n" << pad;
                    size_t stride = aclDataTypeBytes(desc->dtype);
                    for (int d = depth + 1; d < curDims.size(); ++d) stride *= curDims[d];
                    const void* subData = static_cast<const char*>(data) + i * stride;
                    printRec(oss, subData, curDims, depth + 1, indent);
                }
            }
            oss << ']';
        }
    };

    printRec(oss, buf->data, dims, 0, baseIndent);
    return oss.str();
}

enum TensorPrintFlags {
    PRINT_DESC = 1 << 0,
    PRINT_DATA = 1 << 1,
    PRINT_ALL  = PRINT_DESC | PRINT_DATA
};

static std::string formatTensorList(const char* label,
                                    const aclTensorDesc* const descs[],
                                    const aclDataBuffer* const bufs[],
                                    int count,
                                    int flags = PRINT_ALL) {
    std::ostringstream oss;
    for (int i = 0; i < count; ++i) {
        if (descs[i]) {
            if (flags & PRINT_DESC)
                oss << "\n    " << label << "[" << i << "]: "
                    << tensorDescToString(descs[i]);
            if (flags & PRINT_DATA) {
                if (bufs[i] && bufs[i]->data)
                    oss << '\n' << tensorDataToString(descs[i], bufs[i]);
                else
                    oss << "\n    (no buffer)";
            }
        } else
            oss << "\nnull";
    }
    return oss.str();
}


// ~~~ реестр операций ~~~

typedef enum {
    H_UNKNOWN_OP    = -1,
    H_OK             = 0,
    H_UNASSERTED     = 1,
    H_UNIMPLEMENTED  = 2,
} exitCode;

using OpHandler = exitCode (*)(int numInputs, const aclTensorDesc* const inputDesc[],
                               const aclDataBuffer* const inputs[],
                               int numOutputs, const aclTensorDesc* const outputDesc[],
                               aclDataBuffer* const outputs[],
                               const aclopAttr* attr);

struct OpRegistry {
    static std::unordered_map<std::string, OpHandler>& map() {
        static std::unordered_map<std::string, OpHandler> m;
        return m;
    }
    static void add(const std::string& name, OpHandler handler) {
        map()[name] = handler;
    }
    static bool try_find(const std::string& name, OpHandler &result) {
        auto it = map().find(name);
        bool found = it != map().end();
        if (found)
            result = it->second;
        return found;
    }
};

#define REGISTER_OP(NAME, BODY) \
    static exitCode _op_##NAME(int numInputs, const aclTensorDesc* const inputDesc[], \
                               const aclDataBuffer* const inputs[], \
                               int numOutputs, const aclTensorDesc* const outputDesc[], \
                               aclDataBuffer* const outputs[], \
                               const aclopAttr* attr) { \
        BODY \
    } \
    static bool _reg_##NAME = (OpRegistry::add(#NAME, _op_##NAME), true)

#define TRY(expr) \
    do { \
        exitCode _code = (expr); \
        if (_code != H_OK) \
            return _code; \
    } while(0);

#define ASSERT_CODE(cond, code) do { if (!(cond)) return (code); } while (0);
#define ASSERT(cond) ASSERT_CODE(cond, H_UNASSERTED)


// базовая помощь в работе операций

static inline at::ScalarType toAtenType(aclDataType dt) {
    switch (dt) {
        case ACL_FLOAT:   return at::kFloat;
        case ACL_DOUBLE:  return at::kDouble;
        case ACL_FLOAT16: return at::kHalf;
        case ACL_BF16:    return at::kBFloat16;
        case ACL_INT8:    return at::kChar;
        case ACL_UINT8:   return at::kByte;
        case ACL_INT16:   return at::kShort;
        case ACL_UINT16:  return at::kUInt16;
        case ACL_INT32:   return at::kInt;
        case ACL_UINT32:  return at::kUInt32;
        case ACL_INT64:   return at::kLong;
        case ACL_UINT64:  return at::kUInt64;
        case ACL_BOOL:    return at::kBool;
        case ACL_COMPLEX64:  return at::kComplexFloat;
        case ACL_COMPLEX128: return at::kComplexDouble;
        case ACL_COMPLEX32:  // нет прямого аналога, fallback к complex float
                             return at::kComplexFloat;
        default:          return at::ScalarType::Undefined;
    }
}

static inline bool try_toAtenType(aclDataType dt, at::ScalarType &type) {
    at::ScalarType test = toAtenType(dt);
    bool found = test != at::ScalarType::Undefined;
    if (found)
        type = test;
    return found;
}

template<typename T>
static inline bool try_get_attr(const aclopAttr* attr, const std::string& key, T& value) {
    if (!attr)
        return false;
    if constexpr (std::is_same_v<T, int>) {
        auto it = attr->ints.find(key);
        if (it == attr->ints.end())
            return false;
        value = static_cast<int>(it->second);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        auto it = attr->ints.find(key);
        if (it == attr->ints.end())
            return false;
        value = it->second;
    } else if constexpr (std::is_same_v<T, bool>) {
        auto it = attr->bools.find(key);
        if (it == attr->bools.end())
            return false;
        value = it->second;
    } else if constexpr (std::is_same_v<T, float>) {
        auto it = attr->floats.find(key);
        if (it == attr->floats.end())
            return false;
        value = it->second;
    } else if constexpr (std::is_same_v<T, std::string>) {
        auto it = attr->strings.find(key);
        if (it == attr->strings.end())
            return false;
        value = it->second;
    } else if constexpr (std::is_same_v<T, aclDataType>) {
        auto it = attr->dtypes.find(key);
        if (it == attr->dtypes.end())
            return false;
        value = it->second;
    } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
        auto it = attr->list_ints.find(key);
        if (it == attr->list_ints.end())
            return false;
        value = it->second;
    } else if constexpr (std::is_same_v<T, std::vector<uint8_t>>) {
        auto it = attr->list_bools.find(key);
        if (it == attr->list_bools.end())
            return false;
        value = it->second;
    } else if constexpr (std::is_same_v<T, std::vector<float>>) {
        auto it = attr->list_floats.find(key);
        if (it == attr->list_floats.end())
            return false;
        value = it->second;
    } else if constexpr (std::is_same_v<T, std::vector<std::vector<int64_t>>>) {
        auto it = attr->list_list_ints.find(key);
        if (it == attr->list_list_ints.end())
            return false;
        value = it->second;
    } else
        static_assert(sizeof(T) == 0, "Unsupported attribute type");
    return true;
}

static inline exitCode toAtenTensor(const aclTensorDesc* desc, const aclDataBuffer* buffer, at::Tensor& tensor) {
    auto tensor_dims = desc->dims;
    at::ScalarType type;
    if (try_toAtenType(desc->dtype, type)) {
        auto opts = at::TensorOptions().dtype(type).device(at::kCPU);
        tensor = at::from_blob(buffer->data, tensor_dims, opts);
        return H_OK;
    } else
        return H_UNIMPLEMENTED;
}

template <int N>
static inline exitCode toAtenTensors(const aclTensorDesc* const inputDesc[], const aclDataBuffer* const inputs[], at::Tensor (&tensors)[N]) {
    for (int i = 0; i < N; ++i) {
        if (!inputs[i] || !inputDesc[i] || !inputs[i]->data)
            return H_UNASSERTED;
        TRY(toAtenTensor(inputDesc[i], inputs[i], tensors[i]));
    }
    return H_OK;
}

static inline exitCode toAtenTensors(int N, const aclTensorDesc* const inputDesc[], const aclDataBuffer* const inputs[], std::vector<at::Tensor>& tensors) {
    tensors.clear();
    tensors.reserve(N);
    for (int i = 0; i < N; ++i) {
        if (!inputs[i] || !inputDesc[i] || !inputs[i]->data)
            return H_UNASSERTED;
        at::Tensor t;
        TRY(toAtenTensor(inputDesc[i], inputs[i], t));
        tensors.push_back(std::move(t));
    }
    return H_OK;
}


// ~~~ traits типов данных ~~~

// Прямое отображение aclDataType -> C++ тип
template <aclDataType DT> struct aclDataTypeTraits;

#define DEFINE_ACL_TYPE(dt, cpp_type, to_float_expr, from_float_expr) \
    template <> struct aclDataTypeTraits<dt> { \
        using type = cpp_type; \
        static float to_float(const type& v) { return to_float_expr; } \
        static type from_float(float f) { return from_float_expr; } \
    };

// Основные типы
DEFINE_ACL_TYPE(ACL_FLOAT,   float,       v,       f)
DEFINE_ACL_TYPE(ACL_DOUBLE,  double,      static_cast<float>(v), static_cast<double>(f))
DEFINE_ACL_TYPE(ACL_INT8,    int8_t,      static_cast<float>(v), static_cast<int8_t>(f))
DEFINE_ACL_TYPE(ACL_UINT8,   uint8_t,     static_cast<float>(v), static_cast<uint8_t>(f))
DEFINE_ACL_TYPE(ACL_INT16,   int16_t,     static_cast<float>(v), static_cast<int16_t>(f))
DEFINE_ACL_TYPE(ACL_UINT16,  uint16_t,    static_cast<float>(v), static_cast<uint16_t>(f))
DEFINE_ACL_TYPE(ACL_INT32,   int32_t,     static_cast<float>(v), static_cast<int32_t>(f))
DEFINE_ACL_TYPE(ACL_UINT32,  uint32_t,    static_cast<float>(v), static_cast<uint32_t>(f))
DEFINE_ACL_TYPE(ACL_INT64,   int64_t,     static_cast<float>(v), static_cast<int64_t>(f))
DEFINE_ACL_TYPE(ACL_UINT64,  uint64_t,    static_cast<float>(v), static_cast<uint64_t>(f))
DEFINE_ACL_TYPE(ACL_BOOL,    bool,        v ? 1.0f : 0.0f, f != 0.0f)

// Для float16/bf16 и экзотических типов – храним как uint16_t/uint8_t,
// но арифметика через float
inline float half_to_float(uint16_t v) {
    uint32_t bits = static_cast<uint32_t>(v) << 16;
    float res;
    std::memcpy(&res, &bits, sizeof(res));
    return res;
}
inline uint16_t float_to_half(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}
// bf16 имеет аналогичное бинарное представление (старшие 16 бит float)
inline float bf16_to_float(uint16_t v) {
    uint32_t bits = static_cast<uint32_t>(v) << 16;
    float res;
    std::memcpy(&res, &bits, sizeof(res));
    return res;
}
inline uint16_t float_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

DEFINE_ACL_TYPE(ACL_FLOAT16, uint16_t, half_to_float(v), float_to_half(f))
DEFINE_ACL_TYPE(ACL_BF16,    uint16_t, bf16_to_float(v), float_to_bf16(f))

// Экзотические форматы: храним как uint8_t, операции через float
#define DEFINE_EXOTIC_TYPE(dt) \
    DEFINE_ACL_TYPE(dt, uint8_t, static_cast<float>(v), static_cast<uint8_t>(f))

DEFINE_EXOTIC_TYPE(ACL_INT4)
DEFINE_EXOTIC_TYPE(ACL_UINT1)
DEFINE_EXOTIC_TYPE(ACL_COMPLEX32)   // не совсем корректно, но для заглушки сойдёт
DEFINE_EXOTIC_TYPE(ACL_COMPLEX64)
DEFINE_EXOTIC_TYPE(ACL_COMPLEX128)
DEFINE_EXOTIC_TYPE(ACL_HIFLOAT8)
DEFINE_EXOTIC_TYPE(ACL_FLOAT8_E5M2)
DEFINE_EXOTIC_TYPE(ACL_FLOAT8_E4M3FN)
DEFINE_EXOTIC_TYPE(ACL_FLOAT8_E8M0)
DEFINE_EXOTIC_TYPE(ACL_FLOAT6_E3M2)
DEFINE_EXOTIC_TYPE(ACL_FLOAT6_E2M3)
DEFINE_EXOTIC_TYPE(ACL_FLOAT4_E2M1)
DEFINE_EXOTIC_TYPE(ACL_FLOAT4_E1M2)
DEFINE_EXOTIC_TYPE(ACL_HIFLOAT4)
DEFINE_EXOTIC_TYPE(ACL_STRING)      // не используется, но пусть будет
DEFINE_EXOTIC_TYPE(ACL_DT_UNDEFINED)

#undef DEFINE_ACL_TYPE
#undef DEFINE_EXOTIC_TYPE


template <typename T>
class TensorAccessor {
    const T* data_;
    std::vector<int64_t> dims_;
    size_t totalElements_;
public:
    TensorAccessor(const void* data, const std::vector<int64_t>& dims)
        : data_(static_cast<const T*>(data)), dims_(dims), totalElements_(1)
    {
        for (auto d : dims_) totalElements_ *= d;
    }

    size_t numElements() const { return totalElements_; }
    const std::vector<int64_t>& dims() const { return dims_; }

    // Доступ по линейному индексу (константный)
    const T& operator[](size_t i) const { return data_[i]; }
    // Доступ для записи (если нужно модифицировать выход)
    T& operator[](size_t i) { return const_cast<T*>(data_)[i]; }

    // Итераторы (для удобства)
    const T* begin() const { return data_; }
    const T* end() const { return data_ + totalElements_; }
};

// Создание аксессора по aclDataType и буферу
template <aclDataType DT>
auto makeTensorAccessor(const void* data, const std::vector<int64_t>& dims) {
    using T = typename aclDataTypeTraits<DT>::type;
    return TensorAccessor<T>(data, dims);
}


template <typename T, typename BinaryOp>
void broadcastBinaryOp(TensorAccessor<T>& out,
                       const TensorAccessor<T>& a,
                       const std::vector<int64_t>& aDims,
                       const TensorAccessor<T>& b,
                       const std::vector<int64_t>& bDims,
                       BinaryOp op) {
    // Скалярный случай
    if (out.numElements() == 1 && a.numElements() == 1 && b.numElements() == 1) {
        out[0] = op(a[0], b[0]);
        return;
    }

    const auto& outDims = out.dims();
    const size_t ndim = outDims.size();

    auto expandDims = [ndim](const std::vector<int64_t>& dims) {
        std::vector<int64_t> expanded(ndim, 1);
        size_t offset = ndim - dims.size();
        std::copy(dims.begin(), dims.end(), expanded.begin() + offset);
        return expanded;
    };
    std::vector<int64_t> aBroad = expandDims(aDims);
    std::vector<int64_t> bBroad = expandDims(bDims);

    auto computeStrides = [](const std::vector<int64_t>& dims) {
        std::vector<int64_t> strides(dims.size());
        if (!dims.empty()) {
            strides.back() = 1;
            for (int i = dims.size()-2; i >= 0; --i)
                strides[i] = strides[i+1] * dims[i];
        }
        return strides;
    };
    std::vector<int64_t> outStrides = computeStrides(outDims);
    std::vector<int64_t> aStrides = computeStrides(aBroad);
    std::vector<int64_t> bStrides = computeStrides(bBroad);

    const size_t outSize = out.numElements();
    for (size_t linear = 0; linear < outSize; ++linear) {
        size_t aIdx = 0, bIdx = 0;
        size_t temp = linear;
        for (size_t d = 0; d < ndim; ++d) {
            int64_t coord = temp / outStrides[d];
            temp %= outStrides[d];
            if (aBroad[d] > 1) aIdx += coord * aStrides[d];
            if (bBroad[d] > 1) bIdx += coord * bStrides[d];
        }
        out[linear] = op(a[aIdx], b[bIdx]);
    }
}


template <aclDataType DT, typename BinaryPredicate>
void broadcastCompareOp(TensorAccessor<bool>& out,
                        const TensorAccessor<typename aclDataTypeTraits<DT>::type>& a,
                        const std::vector<int64_t>& aDims,
                        const TensorAccessor<typename aclDataTypeTraits<DT>::type>& b,
                        const std::vector<int64_t>& bDims,
                        BinaryPredicate pred) {
    using T = typename aclDataTypeTraits<DT>::type;
    if (out.numElements() == 1 && a.numElements() == 1 && b.numElements() == 1) {
        // Скалярный случай
        out[0] = pred(a[0], b[0]);
        return;
    }

    const auto& outDims = out.dims();
    const size_t ndim = outDims.size();

    // расширяем размерности
    auto expandDims = [ndim](const std::vector<int64_t>& dims) {
        std::vector<int64_t> expanded(ndim, 1);
        size_t offset = ndim - dims.size();
        std::copy(dims.begin(), dims.end(), expanded.begin() + offset);
        return expanded;
    };
    std::vector<int64_t> aBroad = expandDims(aDims);
    std::vector<int64_t> bBroad = expandDims(bDims);

    auto computeStrides = [](const std::vector<int64_t>& dims) {
        std::vector<int64_t> strides(dims.size());
        strides.back() = 1;
        for (int i = dims.size()-2; i >= 0; --i)
            strides[i] = strides[i+1] * dims[i];
        return strides;
    };
    std::vector<int64_t> outStrides = computeStrides(outDims);
    std::vector<int64_t> aStrides = computeStrides(aBroad);
    std::vector<int64_t> bStrides = computeStrides(bBroad);

    const size_t outSize = out.numElements();
    for (size_t linear = 0; linear < outSize; ++linear) {
        size_t aIdx = 0, bIdx = 0;
        size_t temp = linear;
        for (size_t d = 0; d < ndim; ++d) {
            int64_t coord = temp / outStrides[d];
            temp %= outStrides[d];
            if (aBroad[d] > 1) aIdx += coord * aStrides[d];
            if (bBroad[d] > 1) bIdx += coord * bStrides[d];
        }
        out[linear] = pred(a[aIdx], b[bIdx]);
    }
}


template <aclDataType DT>
void fillRandomNormal(TensorAccessor<typename aclDataTypeTraits<DT>::type>& out, std::mt19937& rng) {
    using T = typename aclDataTypeTraits<DT>::type;
    const size_t count = out.numElements();
    if constexpr (DT == ACL_BOOL) {
        std::bernoulli_distribution dist(0.5);
        for (size_t i = 0; i < count; ++i)
            out[i] = dist(rng);
    } else if constexpr (std::is_floating_point_v<T> || DT == ACL_FLOAT16 || DT == ACL_BF16) {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < count; ++i) {
            float v = dist(rng);
            out[i] = aclDataTypeTraits<DT>::from_float(v);
        }
    } else if constexpr (std::is_integral_v<T>) {
        using limits = std::numeric_limits<T>;
        if constexpr (std::is_signed_v<T>) {
            std::uniform_int_distribution<long long> dist(limits::min(), limits::max());
            for (size_t i = 0; i < count; ++i)
                out[i] = static_cast<T>(dist(rng));
        } else {
            std::uniform_int_distribution<unsigned long long> dist(limits::min(), limits::max());
            for (size_t i = 0; i < count; ++i)
                out[i] = static_cast<T>(dist(rng));
        }
    } else {
        // fallback: заполняем нулями
        std::memset(&out[0], 0, count * sizeof(T));
    }
}


#define DISPATCH_RANDOM(DT) \
    case DT: { \
        using T = aclDataTypeTraits<DT>::type; \
        TensorAccessor<T> out(outputs[0]->data, outputDesc[0]->dims); \
        fillRandomNormal<DT>(out, local_rng); \
        break; \
    }

#define DISPATCH_RANDOM_UNIFORM(DT) \
    case DT: { \
        using T = aclDataTypeTraits<DT>::type; \
        TensorAccessor<T> out(outputs[0]->data, outputDesc[0]->dims); \
        if constexpr (std::is_floating_point_v<T> || DT == ACL_FLOAT16 || DT == ACL_BF16) { \
            std::uniform_real_distribution<float> dist(0.0f, 1.0f); \
            for (size_t j = 0; j < out.numElements(); ++j) \
                out[j] = aclDataTypeTraits<DT>::from_float(dist(local_rng)); \
        } else if constexpr (DT == ACL_BOOL) { \
            std::bernoulli_distribution dist(0.5); \
            for (size_t j = 0; j < out.numElements(); ++j) \
                out[j] = dist(local_rng); \
        } else { \
            using limits = std::numeric_limits<T>; \
            if constexpr (std::is_signed_v<T>) { \
                std::uniform_int_distribution<long long> dist(limits::min(), limits::max()); \
                for (size_t j = 0; j < out.numElements(); ++j) \
                    out[j] = static_cast<T>(dist(local_rng)); \
            } else { \
                std::uniform_int_distribution<unsigned long long> dist(limits::min(), limits::max()); \
                for (size_t j = 0; j < out.numElements(); ++j) \
                    out[j] = static_cast<T>(dist(local_rng)); \
            } \
        } \
        break; \
    }


#define DISPATCH_ZEROS_LIKE(DT) \
    case DT: { \
        using T = aclDataTypeTraits<DT>::type; \
        TensorAccessor<T> out(outputs[0]->data, outputDesc[0]->dims); \
        size_t count = out.numElements(); \
        if constexpr (DT == ACL_BOOL) { \
            for (size_t i = 0; i < count; ++i) out[i] = false; \
        } else { \
            for (size_t i = 0; i < count; ++i) out[i] = static_cast<T>(0); \
        } \
        break; \
    }

#define DISPATCH_FILL(DT) \
    case DT: { \
        using T = aclDataTypeTraits<DT>::type; \
        TensorAccessor<T> out(outputs[0]->data, outputDesc[0]->dims); \
        TensorAccessor<T> value(inputs[1]->data, inputDesc[1]->dims); \
        T fillValue = value[0]; /* второй вход – скаляр */ \
        size_t count = out.numElements(); \
        for (size_t i = 0; i < count; ++i) out[i] = fillValue; \
        break; \
    }


#define DISPATCH_MUL(DT) \
    case DT: { \
        using T = aclDataTypeTraits<DT>::type; \
        TensorAccessor<T> out(outputs[0]->data, outputDesc[0]->dims); \
        TensorAccessor<T> inA(inputs[0]->data, inputDesc[0]->dims); \
        TensorAccessor<T> inB(inputs[1]->data, inputDesc[1]->dims); \
        broadcastBinaryOp(out, inA, inputDesc[0]->dims, \
                            inB, inputDesc[1]->dims, \
                            std::multiplies<T>{}); \
        break; \
    }

#define DISPATCH_ADD(DT) \
    case DT: { \
        using T = aclDataTypeTraits<DT>::type; \
        TensorAccessor<T> out(outputs[0]->data, outputDesc[0]->dims); \
        TensorAccessor<T> inA(inputs[0]->data, inputDesc[0]->dims); \
        TensorAccessor<T> inB(inputs[1]->data, inputDesc[1]->dims); \
        broadcastBinaryOp(out, inA, inputDesc[0]->dims, \
                            inB, inputDesc[1]->dims, \
                            std::plus<T>{}); \
        break; \
    }

#define DISPATCH_SUB(DT) \
    case DT: { \
        using T = aclDataTypeTraits<DT>::type; \
        TensorAccessor<T> out(outputs[0]->data, outputDesc[0]->dims); \
        TensorAccessor<T> inA(inputs[0]->data, inputDesc[0]->dims); \
        TensorAccessor<T> inB(inputs[1]->data, inputDesc[1]->dims); \
        broadcastBinaryOp(out, inA, inputDesc[0]->dims, \
                            inB, inputDesc[1]->dims, \
                            std::minus<T>{}); \
        break; \
    }

#define DISPATCH_DIV(DT) \
    case DT: { \
        using T = aclDataTypeTraits<DT>::type; \
        TensorAccessor<T> out(outputs[0]->data, outputDesc[0]->dims); \
        TensorAccessor<T> inA(inputs[0]->data, inputDesc[0]->dims); \
        TensorAccessor<T> inB(inputs[1]->data, inputDesc[1]->dims); \
        auto divOp = [](T a, T b) -> T { \
            if constexpr (std::is_integral_v<T>) { \
                if (b == 0) return 0; \
                if constexpr (std::is_signed_v<T>) { \
                    if (b == -1 && a == std::numeric_limits<T>::min()) return 0; \
                } \
                return a / b; \
            } else { \
                return a / b; \
            } \
        }; \
        broadcastBinaryOp(out, inA, inputDesc[0]->dims, \
                          inB, inputDesc[1]->dims, divOp); \
        break; \
    }


#define DISPATCH_ISFINITE(DT) \
    case DT: { \
        using T = aclDataTypeTraits<DT>::type; \
        TensorAccessor<T> in(inputs[0]->data, inputDesc[0]->dims); \
        TensorAccessor<bool> out(outputs[0]->data, outputDesc[0]->dims); \
        size_t count = out.numElements(); \
        for (size_t i = 0; i < count; ++i) { \
            if constexpr (std::is_floating_point_v<T> || \
                          DT == ACL_FLOAT16 || DT == ACL_BF16) { \
                float val = aclDataTypeTraits<DT>::to_float(in[i]); \
                out[i] = std::isfinite(val); \
            } else if constexpr (std::is_integral_v<T> || DT == ACL_BOOL) { \
                out[i] = true; \
            } else { \
                out[i] = false; /* fallback */ \
            } \
        } \
        break; \
    }

#define DISPATCH_COMPARE(DT, PRED) \
    case DT: { \
        using T = aclDataTypeTraits<DT>::type; \
        TensorAccessor<T> inA(inputs[0]->data, inputDesc[0]->dims); \
        TensorAccessor<T> inB(inputs[1]->data, inputDesc[1]->dims); \
        TensorAccessor<bool> out(outputs[0]->data, outputDesc[0]->dims); \
        broadcastCompareOp<DT>(out, inA, inputDesc[0]->dims, \
                               inB, inputDesc[1]->dims, PRED); \
        break; \
    }

#define DISPATCH_TENSOR_EQUAL(DT) \
    case DT: { \
        using T = aclDataTypeTraits<DT>::type; \
        const T* p0 = static_cast<const T*>(inputs[0]->data); \
        const T* p1 = static_cast<const T*>(inputs[1]->data); \
        bool all_equal = true; \
        size_t count = calc_num_elements(inputDesc[0], inputs[0]->size); \
        for (size_t i = 0; i < count; ++i) { \
            if constexpr (DT == ACL_FLOAT16) { \
                if (half_to_float(p0[i]) != half_to_float(p1[i])) { all_equal = false; break; } \
            } else if constexpr (DT == ACL_BF16) { \
                if (bf16_to_float(p0[i]) != bf16_to_float(p1[i])) { all_equal = false; break; } \
            } else { \
                if (p0[i] != p1[i]) { all_equal = false; break; } \
            } \
        } \
        *static_cast<bool*>(outputs[0]->data) = all_equal; \
        break; \
    }

#define DISPATCH_MASKED_FILL(DT) \
    case DT: { \
        using T = aclDataTypeTraits<DT>::type; \
        TensorAccessor<T> in(inputs[0]->data, inputDesc[0]->dims); \
        TensorAccessor<bool> mask(inputs[1]->data, inputDesc[1]->dims); \
        TensorAccessor<T> out(outputs[0]->data, outputDesc[0]->dims); \
        /* value может быть либо скалярным тензором, либо тензором той же формы */ \
        T fillValue; \
        if (inputDesc[2]->dims.empty() || calc_num_elements(inputDesc[2], inputs[2]->size) == 1) { \
            fillValue = static_cast<const T*>(inputs[2]->data)[0]; \
        } else { \
            /* Тензор значений – будем брать из value по индексу (broadcast не делаем, обычно совпадает по форме с self) */ \
            fillValue = static_cast<const T*>(inputs[2]->data)[0]; /* временное упрощение */ \
        } \
        size_t total = in.numElements(); \
        for (size_t i = 0; i < total; ++i) { \
            out[i] = mask[i] ? fillValue : in[i]; \
        } \
        break; \
    }

#define DISPATCH_MASKED_SELECT(DT) \
    case DT: { \
        using T = aclDataTypeTraits<DT>::type; \
        TensorAccessor<T> in(inputs[0]->data, inputDesc[0]->dims); \
        TensorAccessor<bool> mask(inputs[1]->data, inputDesc[1]->dims); \
        TensorAccessor<T> out(outputs[0]->data, outputDesc[0]->dims); \
        size_t out_count = 0; \
        size_t total = in.numElements(); \
        for (size_t i = 0; i < total; ++i) { \
            if (mask[i]) { \
                if (out_count < out.numElements()) { \
                    out[out_count] = in[i]; \
                } \
                out_count++; \
            } \
        } \
        break; \
    }

#define DISPATCH_UNARY(DT, OP) \
    case DT: { \
        using T = aclDataTypeTraits<DT>::type; \
        TensorAccessor<T> in(inputs[0]->data, inputDesc[0]->dims); \
        TensorAccessor<T> out(outputs[0]->data, outputDesc[0]->dims); \
        size_t count = in.numElements(); \
        for (size_t i = 0; i < count; ++i) { \
            out[i] = OP(in[i]); \
        } \
        break; \
    }

#define DISPATCH_REDUCE(DT, INIT, OP) \
    case DT: { \
        using T = aclDataTypeTraits<DT>::type; \
        TensorAccessor<T> in(inputs[0]->data, inputDesc[0]->dims); \
        TensorAccessor<T> out(outputs[0]->data, outputDesc[0]->dims); \
        if (num_axes == 0) { \
            T acc = INIT; \
            for (size_t i = 0; i < in.numElements(); ++i) acc = OP(acc, in[i]); \
            out[0] = acc; \
            break; \
        } \
        const int64_t* axes = static_cast<const int64_t*>(inputs[1]->data); \
        size_t ndim = in.dims().size(); \
        /* Строим маску редуцируемых осей и список осей для обхода */ \
        std::vector<bool> reduce_mask(ndim, false); \
        for (size_t i = 0; i < num_axes; ++i) { \
            int64_t ax = axes[i]; \
            if (ax < 0) ax += ndim; \
            reduce_mask[ax] = true; \
        } \
        /* Вычисляем strides для входного тензора */ \
        std::vector<int64_t> in_strides(ndim, 1); \
        for (int i = ndim - 2; i >= 0; --i) \
            in_strides[i] = in_strides[i+1] * in.dims()[i+1]; \
        /* Формируем выходные размеры (сохраняя нередуцируемые оси) */ \
        std::vector<int64_t> out_dims; \
        for (size_t d = 0; d < ndim; ++d) \
            if (!reduce_mask[d]) out_dims.push_back(in.dims()[d]); \
        if (out_dims.empty()) out_dims.push_back(1); \
        size_t out_total = 1; \
        for (auto d : out_dims) out_total *= d; \
        /* Для каждого выходного элемента */ \
        for (size_t out_idx = 0; out_idx < out_total; ++out_idx) { \
            /* Восстанавливаем координаты в выходном пространстве */ \
            std::vector<int64_t> out_coord(out_dims.size(), 0); \
            size_t tmp = out_idx; \
            for (int i = out_dims.size()-1; i >= 0; --i) { \
                out_coord[i] = tmp % out_dims[i]; \
                tmp /= out_dims[i]; \
            } \
            /* Строим базовые координаты входа (нередуцируемые оси фиксированы) */ \
            std::vector<int64_t> base_coord(ndim, 0); \
            size_t out_dim_pos = 0; \
            for (size_t d = 0; d < ndim; ++d) { \
                if (!reduce_mask[d]) base_coord[d] = out_coord[out_dim_pos++]; \
            } \
            /* Вычисляем линейный индекс базового элемента */ \
            size_t base_idx = 0; \
            for (size_t d = 0; d < ndim; ++d) \
                base_idx += base_coord[d] * in_strides[d]; \
            T acc = INIT; \
            /* Теперь итеративно перебираем все комбинации редуцируемых осей */ \
            /* Используем массив текущих координат вдоль редуцируемых осей и их strides */ \
            std::vector<int64_t> red_axes, red_sizes, red_strides; \
            for (size_t d = 0; d < ndim; ++d) { \
                if (reduce_mask[d]) { \
                    red_axes.push_back(d); \
                    red_sizes.push_back(in.dims()[d]); \
                    red_strides.push_back(in_strides[d]); \
                } \
            } \
            size_t num_red = red_axes.size(); \
            if (num_red == 0) { \
                /* Нет редуцируемых осей – просто копируем значение */ \
                acc = in[base_idx]; \
            } else { \
                /* Перебираем все комбинации через линейный счётчик */ \
                size_t total_red = 1; \
                for (auto sz : red_sizes) total_red *= sz; \
                for (size_t combo = 0; combo < total_red; ++combo) { \
                    size_t idx = base_idx; \
                    size_t rem = combo; \
                    for (size_t r = 0; r < num_red; ++r) { \
                        int64_t coord = rem % red_sizes[r]; \
                        rem /= red_sizes[r]; \
                        idx += coord * red_strides[r]; \
                    } \
                    acc = OP(acc, in[idx]); \
                } \
            } \
            out[out_idx] = acc; \
        } \
        break; \
    }

#define DISPATCH_REDUCE_MAX(DT)  DISPATCH_REDUCE(DT, std::numeric_limits<T>::lowest(), [](T a, T b) { return a > b ? a : b; })
#define DISPATCH_REDUCE_MIN(DT)  DISPATCH_REDUCE(DT, std::numeric_limits<T>::max(),    [](T a, T b) { return a < b ? a : b; })
#define DISPATCH_REDUCE_ALL(DT)  DISPATCH_REDUCE(DT, true,  [](T a, T b) { return (a != static_cast<T>(0)) && (b != static_cast<T>(0)); })
#define DISPATCH_REDUCE_ANY(DT)  DISPATCH_REDUCE(DT, false, [](T a, T b) { return (a != static_cast<T>(0)) || (b != static_cast<T>(0)); })
#define DISPATCH_REDUCE_SUM(DT)  DISPATCH_REDUCE(DT, static_cast<T>(0),                [](T a, T b) { return a + b; })
#define DISPATCH_REDUCE_PROD(DT) DISPATCH_REDUCE(DT, static_cast<T>(1),                [](T a, T b) { return a * b; })

#define DISPATCH_ARG_MAX_WITH_VALUE(DT) \
    case DT: { \
        using T = aclDataTypeTraits<DT>::type; \
        TensorAccessor<T> in(inputs[0]->data, inputDesc[0]->dims); \
        /* ВНИМАНИЕ: outputs[0] — ИНДЕКСЫ (int32), outputs[1] — ЗНАЧЕНИЯ (T) */ \
        TensorAccessor<int32_t> out_idx(outputs[0]->data, outputDesc[0]->dims); \
        TensorAccessor<T> out_vals(outputs[1]->data, outputDesc[1]->dims); \
        \
        const auto& dim_attr = attr->ints.find("dimension"); \
        if (dim_attr == attr->ints.end()) return H_UNASSERTED; \
        int64_t axis = dim_attr->second; \
        if (axis < 0) axis += in.dims().size(); \
        \
        bool keepdim = false; \
        auto keep_attr = attr->bools.find("keep_dims"); \
        if (keep_attr != attr->bools.end()) keepdim = keep_attr->second; \
        \
        std::vector<int64_t> outDims = in.dims(); \
        outDims.erase(outDims.begin() + axis); \
        size_t outSize = out_vals.numElements(); \
        \
        for (size_t outIdx = 0; outIdx < outSize; ++outIdx) { \
            size_t remaining = outIdx; \
            std::vector<int64_t> coord(outDims.size(), 0); \
            for (int i = outDims.size()-1; i >= 0; --i) { \
                coord[i] = remaining % outDims[i]; \
                remaining /= outDims[i]; \
            } \
            std::vector<int64_t> fullCoord = coord; \
            fullCoord.insert(fullCoord.begin() + axis, 0); \
            T max_val = std::numeric_limits<T>::lowest(); \
            int32_t max_idx = 0; \
            for (fullCoord[axis] = 0; fullCoord[axis] < in.dims()[axis]; ++fullCoord[axis]) { \
                size_t inIdx = 0; \
                size_t stride = 1; \
                for (int i = in.dims().size()-1; i >= 0; --i) { \
                    inIdx += fullCoord[i] * stride; \
                    stride *= in.dims()[i]; \
                } \
                T val = in[inIdx]; \
                if (val > max_val) { \
                    max_val = val; \
                    max_idx = static_cast<int32_t>(fullCoord[axis]); \
                } \
            } \
            out_vals[outIdx] = max_val; \
            out_idx[outIdx] = max_idx; \
        } \
        break; \
    }

#define DISPATCH_ARG_MIN_WITH_VALUE(DT) \
    case DT: { \
        using T = aclDataTypeTraits<DT>::type; \
        TensorAccessor<T> in(inputs[0]->data, inputDesc[0]->dims); \
        /* outputs[0] — ИНДЕКСЫ (int32), outputs[1] — ЗНАЧЕНИЯ (T) */ \
        TensorAccessor<int32_t> out_idx(outputs[0]->data, outputDesc[0]->dims); \
        TensorAccessor<T> out_vals(outputs[1]->data, outputDesc[1]->dims); \
        \
        const auto& dim_attr = attr->ints.find("dimension"); \
        if (dim_attr == attr->ints.end()) return H_UNASSERTED; \
        int64_t axis = dim_attr->second; \
        if (axis < 0) axis += in.dims().size(); \
        \
        bool keepdim = false; \
        auto keep_attr = attr->bools.find("keep_dims"); \
        if (keep_attr != attr->bools.end()) keepdim = keep_attr->second; \
        \
        std::vector<int64_t> outDims = in.dims(); \
        outDims.erase(outDims.begin() + axis); \
        size_t outSize = out_vals.numElements(); \
        \
        for (size_t outIdx = 0; outIdx < outSize; ++outIdx) { \
            size_t remaining = outIdx; \
            std::vector<int64_t> coord(outDims.size(), 0); \
            for (int i = outDims.size()-1; i >= 0; --i) { \
                coord[i] = remaining % outDims[i]; \
                remaining /= outDims[i]; \
            } \
            std::vector<int64_t> fullCoord = coord; \
            fullCoord.insert(fullCoord.begin() + axis, 0); \
            T min_val = std::numeric_limits<T>::max(); \
            int32_t min_idx = 0; \
            for (fullCoord[axis] = 0; fullCoord[axis] < in.dims()[axis]; ++fullCoord[axis]) { \
                size_t inIdx = 0; \
                size_t stride = 1; \
                for (int i = in.dims().size()-1; i >= 0; --i) { \
                    inIdx += fullCoord[i] * stride; \
                    stride *= in.dims()[i]; \
                } \
                T val = in[inIdx]; \
                if (val < min_val) { \
                    min_val = val; \
                    min_idx = static_cast<int32_t>(fullCoord[axis]); \
                } \
            } \
            out_vals[outIdx] = min_val; \
            out_idx[outIdx] = min_idx; \
        } \
        break; \
    }

#define DISPATCH_REDUCE_MEAN(DT) \
    case DT: { \
        using T = aclDataTypeTraits<DT>::type; \
        TensorAccessor<T> in(inputs[0]->data, inputDesc[0]->dims); \
        TensorAccessor<T> out(outputs[0]->data, outputDesc[0]->dims); \
        if (num_axes == 0) { \
            T sum = 0; \
            for (size_t i = 0; i < in.numElements(); ++i) sum += in[i]; \
            out[0] = (in.numElements() > 0) ? sum / static_cast<T>(in.numElements()) : 0; \
            break; \
        } \
        const int64_t* axes = static_cast<const int64_t*>(inputs[1]->data); \
        size_t ndim = in.dims().size(); \
        std::vector<bool> reduce_mask(ndim, false); \
        size_t reduce_count = 1; \
        for (size_t i = 0; i < num_axes; ++i) { \
            int64_t ax = axes[i]; \
            if (ax < 0) ax += ndim; \
            reduce_mask[ax] = true; \
            reduce_count *= in.dims()[ax]; \
        } \
        std::vector<int64_t> in_strides(ndim, 1); \
        for (int i = ndim - 2; i >= 0; --i) \
            in_strides[i] = in_strides[i+1] * in.dims()[i+1]; \
        std::vector<int64_t> keep_dims; \
        for (size_t d = 0; d < ndim; ++d) \
            if (!reduce_mask[d]) keep_dims.push_back(in.dims()[d]); \
        if (keep_dims.empty()) keep_dims.push_back(1); \
        size_t out_total = 1; \
        for (auto d : keep_dims) out_total *= d; \
        for (size_t out_idx = 0; out_idx < out_total; ++out_idx) { \
            std::vector<int64_t> keep_coord(keep_dims.size(), 0); \
            size_t temp = out_idx; \
            for (int i = keep_dims.size()-1; i >= 0; --i) { \
                keep_coord[i] = temp % keep_dims[i]; \
                temp /= keep_dims[i]; \
            } \
            std::vector<int64_t> cur_coord(ndim, 0); \
            size_t keep_idx = 0; \
            for (size_t d = 0; d < ndim; ++d) { \
                if (!reduce_mask[d]) cur_coord[d] = keep_coord[keep_idx++]; \
            } \
            T sum = 0; \
            while (true) { \
                size_t in_idx = 0; \
                for (size_t d = 0; d < ndim; ++d) in_idx += cur_coord[d] * in_strides[d]; \
                sum += in[in_idx]; \
                int inc_dim = ndim - 1; \
                while (inc_dim >= 0 && !reduce_mask[inc_dim]) --inc_dim; \
                if (inc_dim < 0) break; \
                while (inc_dim >= 0) { \
                    if (reduce_mask[inc_dim]) { \
                        cur_coord[inc_dim]++; \
                        if (cur_coord[inc_dim] >= in.dims()[inc_dim]) { \
                            cur_coord[inc_dim] = 0; \
                            --inc_dim; \
                        } else break; \
                    } else --inc_dim; \
                } \
                if (inc_dim < 0) break; \
            } \
            out[out_idx] = sum / static_cast<T>(reduce_count); \
        } \
        break; \
    }

// ReduceMeanD: оси передаются через атрибут "axes" (list_int), keep_dims через "keep_dims" (bool)
#define DISPATCH_REDUCE_MEAN_FROM_AXES(DT) \
    case DT: { \
        using T = aclDataTypeTraits<DT>::type; \
        TensorAccessor<T> in(inputs[0]->data, inputDesc[0]->dims); \
        TensorAccessor<T> out(outputs[0]->data, outputDesc[0]->dims); \
        \
        /* Извлекаем оси из атрибутов */ \
        const auto& axes_attr = attr->list_ints.find("axes"); \
        if (axes_attr == attr->list_ints.end()) return H_UNASSERTED; \
        const std::vector<int64_t>& axes = axes_attr->second; \
        \
        /* keep_dims (опционально) */ \
        bool keep_dims = false; \
        auto keep_attr = attr->bools.find("keep_dims"); \
        if (keep_attr != attr->bools.end()) keep_dims = keep_attr->second; \
        \
        /* Нормализуем оси */ \
        std::vector<int64_t> norm_axes; \
        int64_t ndim = static_cast<int64_t>(in.dims().size()); \
        for (int64_t ax : axes) { \
            if (ax < 0) ax += ndim; \
            norm_axes.push_back(ax); \
        } \
        if (norm_axes.empty()) { \
            /* все оси */ \
            for (int64_t i = 0; i < ndim; ++i) norm_axes.push_back(i); \
        } \
        \
        /* Считаем количество сворачиваемых элементов для каждой выходной ячейки */ \
        size_t count = 1; \
        for (int64_t ax : norm_axes) count *= in.dims()[ax]; \
        \
        if (norm_axes.size() == static_cast<size_t>(ndim)) { \
            /* Редукция до скаляра */ \
            T sum = 0; \
            for (size_t i = 0; i < in.numElements(); ++i) sum += in[i]; \
            out[0] = static_cast<T>(sum / static_cast<T>(count)); \
        } else { \
            /* Частичная редукция – используем универсальную логику, как в DISPATCH_REDUCE, но с делением на count */ \
            /* Здесь можно скопировать код из DISPATCH_REDUCE_MEAN с небольшой модификацией: */ \
            /* Для простоты обработки возьмём первую ось (обычно одна ось) */ \
            int64_t axis = norm_axes[0]; \
            std::vector<int64_t> outDims = in.dims(); \
            outDims.erase(outDims.begin() + axis); \
            size_t outSize = out.numElements(); \
            for (size_t outIdx = 0; outIdx < outSize; ++outIdx) { \
                size_t remaining = outIdx; \
                std::vector<int64_t> coord(outDims.size(), 0); \
                for (int i = outDims.size()-1; i >= 0; --i) { \
                    coord[i] = remaining % outDims[i]; \
                    remaining /= outDims[i]; \
                } \
                std::vector<int64_t> fullCoord = coord; \
                fullCoord.insert(fullCoord.begin() + axis, 0); \
                T sum = 0; \
                for (fullCoord[axis] = 0; fullCoord[axis] < in.dims()[axis]; ++fullCoord[axis]) { \
                    size_t inIdx = 0; \
                    size_t stride = 1; \
                    for (int i = in.dims().size()-1; i >= 0; --i) { \
                        inIdx += fullCoord[i] * stride; \
                        stride *= in.dims()[i]; \
                    } \
                    sum += in[inIdx]; \
                } \
                out[outIdx] = static_cast<T>(sum / static_cast<T>(count)); \
            } \
        } \
        break; \
    }

// Макрос для ReduceLogSumExp – log(sum(exp(x))) вдоль заданных осей.
// Вход уже должен быть сдвинут на максимум (x - max), поэтому exp(x) <= 1,
// и накопление суммы exp в double безопасно.
#define DISPATCH_REDUCE_LOG_SUM_EXP(DT) \
    case DT: { \
        using T = aclDataTypeTraits<DT>::type; \
        TensorAccessor<T> in(inputs[0]->data, inputDesc[0]->dims); \
        TensorAccessor<T> out(outputs[0]->data, outputDesc[0]->dims); \
        if (num_axes == 0) { \
            /* глобальная редукция */ \
            T max_val = in[0]; \
            for (size_t i = 1; i < in.numElements(); ++i) \
                if (in[i] > max_val) max_val = in[i]; \
            double sum = 0.0; \
            for (size_t i = 0; i < in.numElements(); ++i) \
                sum += std::exp(static_cast<double>(in[i]) - static_cast<double>(max_val)); \
            out[0] = static_cast<T>(std::log(sum) + static_cast<double>(max_val)); \
            break; \
        } \
        const int64_t* axes = static_cast<const int64_t*>(inputs[1]->data); \
        int64_t axis = axes[0]; \
        if (axis < 0) axis += in.dims().size(); \
        size_t ndim = in.dims().size(); \
        std::vector<int64_t> inStrides(ndim, 1); \
        for (int i = ndim - 2; i >= 0; --i) \
            inStrides[i] = inStrides[i+1] * in.dims()[i+1]; \
        std::vector<int64_t> outDims; \
        for (size_t d = 0; d < ndim; ++d) \
            if (d != axis) outDims.push_back(in.dims()[d]); \
        if (outDims.empty()) outDims.push_back(1); \
        size_t outSize = 1; \
        for (size_t d = 0; d < outDims.size(); ++d) outSize *= outDims[d]; \
        for (size_t outIdx = 0; outIdx < outSize; ++outIdx) { \
            std::vector<int64_t> outCoord(outDims.size(), 0); \
            size_t temp = outIdx; \
            for (int i = outDims.size()-1; i >= 0; --i) { \
                outCoord[i] = temp % outDims[i]; \
                temp /= outDims[i]; \
            } \
            std::vector<int64_t> inCoord(ndim, 0); \
            size_t outDimIdx = 0; \
            for (size_t d = 0; d < ndim; ++d) { \
                if (d == axis) inCoord[d] = 0; \
                else inCoord[d] = outCoord[outDimIdx++]; \
            } \
            /* находим максимум вдоль оси */ \
            T max_val = in[0]; /* временно, ниже пересчитаем */ \
            { \
                size_t first_idx = 0; \
                for (size_t d = 0; d < ndim; ++d) first_idx += inCoord[d] * inStrides[d]; \
                max_val = in[first_idx]; \
                for (inCoord[axis] = 1; inCoord[axis] < in.dims()[axis]; ++inCoord[axis]) { \
                    size_t inIdx = 0; \
                    for (size_t d = 0; d < ndim; ++d) inIdx += inCoord[d] * inStrides[d]; \
                    if (in[inIdx] > max_val) max_val = in[inIdx]; \
                } \
            } \
            double sum = 0.0; \
            for (inCoord[axis] = 0; inCoord[axis] < in.dims()[axis]; ++inCoord[axis]) { \
                size_t inIdx = 0; \
                for (size_t d = 0; d < ndim; ++d) inIdx += inCoord[d] * inStrides[d]; \
                sum += std::exp(static_cast<double>(in[inIdx]) - static_cast<double>(max_val)); \
            } \
            out[outIdx] = static_cast<T>(std::log(sum) + static_cast<double>(max_val)); \
        } \
        break; \
    }

// torch.var
#define REDUCE_STD_DT(DT) \
    case DT: { \
        using T = aclDataTypeTraits<DT>::type; \
        TensorAccessor<T> in(inputs[0]->data, inputDesc[0]->dims); \
        TensorAccessor<T> mean_val(inputs[1]->data, inputDesc[1]->dims); \
        TensorAccessor<T> out(outputs[0]->data, outputDesc[0]->dims); \
        if (norm_dims.empty()) { \
            double sum_sq = 0.0; \
            for (size_t i = 0; i < total_elements; ++i) { \
                double diff = static_cast<double>(in[i]) - static_cast<double>(mean_val[i]); \
                sum_sq += diff * diff; \
            } \
            out[0] = static_cast<T>(denom > 0 ? sum_sq / denom : 0); \
        } else { \
            size_t ndim = in.dims().size(); \
            std::vector<bool> reduce_mask(ndim, false); \
            for (int64_t ax : norm_dims) { \
                if (ax < 0) ax += ndim; \
                reduce_mask[ax] = true; \
            } \
            std::vector<int64_t> in_strides(ndim, 1); \
            for (int i = ndim - 2; i >= 0; --i) \
                in_strides[i] = in_strides[i+1] * in.dims()[i+1]; \
            std::vector<int64_t> keep_dims; \
            for (size_t d = 0; d < ndim; ++d) \
                if (!reduce_mask[d]) keep_dims.push_back(in.dims()[d]); \
            if (keep_dims.empty()) keep_dims.push_back(1); \
            size_t out_total = 1; \
            for (auto d : keep_dims) out_total *= d; \
            for (size_t out_idx = 0; out_idx < out_total; ++out_idx) { \
                std::vector<int64_t> keep_coord(keep_dims.size(), 0); \
                size_t temp = out_idx; \
                for (int i = keep_dims.size()-1; i >= 0; --i) { \
                    keep_coord[i] = temp % keep_dims[i]; \
                    temp /= keep_dims[i]; \
                } \
                std::vector<int64_t> cur_coord(ndim, 0); \
                size_t keep_idx = 0; \
                for (size_t d = 0; d < ndim; ++d) { \
                    if (!reduce_mask[d]) cur_coord[d] = keep_coord[keep_idx++]; \
                } \
                double sum_sq = 0.0; \
                while (true) { \
                    size_t in_idx = 0; \
                    for (size_t d = 0; d < ndim; ++d) in_idx += cur_coord[d] * in_strides[d]; \
                    double diff = static_cast<double>(in[in_idx]) - static_cast<double>(mean_val[in_idx]); \
                    sum_sq += diff * diff; \
                    int inc_dim = ndim - 1; \
                    while (inc_dim >= 0 && !reduce_mask[inc_dim]) --inc_dim; \
                    if (inc_dim < 0) break; \
                    while (inc_dim >= 0) { \
                        if (reduce_mask[inc_dim]) { \
                            cur_coord[inc_dim]++; \
                            if (cur_coord[inc_dim] >= in.dims()[inc_dim]) { \
                                cur_coord[inc_dim] = 0; \
                                --inc_dim; \
                            } else break; \
                        } else --inc_dim; \
                    } \
                    if (inc_dim < 0) break; \
                } \
                out[out_idx] = static_cast<T>(denom > 0 ? sum_sq / denom : 0); \
            } \
        } \
        break; \
    }

#define DISPATCH_BROADCAST_TO(DT) \
    case DT: { \
        using T = aclDataTypeTraits<DT>::type; \
        TensorAccessor<T> in(inputs[0]->data, inputDesc[0]->dims); \
        TensorAccessor<T> out(outputs[0]->data, outputDesc[0]->dims); \
        \
        if (in.numElements() == 0 || out.numElements() == 0) break; \
        \
        std::vector<int64_t> inStrides(in.dims().size(), 1); \
        for (int i = in.dims().size()-2; i >= 0; --i) \
            inStrides[i] = inStrides[i+1] * in.dims()[i+1]; \
        \
        std::vector<int64_t> outStrides(out.dims().size(), 1); \
        for (int i = out.dims().size()-2; i >= 0; --i) \
            outStrides[i] = outStrides[i+1] * out.dims()[i+1]; \
        \
        size_t outSize = out.numElements(); \
        for (size_t outIdx = 0; outIdx < outSize; ++outIdx) { \
            size_t inIdx = 0; \
            size_t temp = outIdx; \
            for (int d = out.dims().size()-1; d >= 0; --d) { \
                int64_t coord = temp % out.dims()[d]; \
                temp /= out.dims()[d]; \
                if (d < in.dims().size() && in.dims()[d] > 1) \
                    inIdx += coord * inStrides[d]; \
            } \
            out[outIdx] = in[inIdx]; \
        } \
        break; \
    }


#endif // HELPERS
