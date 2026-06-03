#include "not_acl.cpp"  // aclGetTensorDescDimV2, aclGetTensorDescNumDims, aclGetTensorDescType
#include <random>       // bernoulli_distribution, mt19937, normal_distribution, random_device, uniform_int_distribution

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
    if (!elemPtr) return "?";
    switch (dtype) {
        case ACL_FLOAT:
            return std::to_string(*static_cast<const float*>(elemPtr));
        case ACL_DOUBLE:
            return std::to_string(*static_cast<const double*>(elemPtr));
        case ACL_FLOAT16:
        case ACL_BF16: {
            uint16_t v = *static_cast<const uint16_t*>(elemPtr);
            uint32_t bits = static_cast<uint32_t>(v) << 16;
            float f;
            std::memcpy(&f, &bits, sizeof(f));
            return std::to_string(f);
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

static void printTensorRecursive(std::ostream& os,
                                 const aclTensorDesc* desc,
                                 const void* data,
                                 const std::vector<int64_t>& dims,
                                 size_t depth,
                                 size_t& offset,
                                 size_t maxShow,
                                 bool& truncated,
                                 int baseIndent) {
    if (truncated) return;
    if (depth == dims.size() - 1) {
        // Последнее измерение – выводим строку элементов
        os << "[";
        size_t dimLen = dims[depth];
        size_t showCount = std::min(dimLen, maxShow - offset);
        if (showCount == 0) { os << "..."; return; }
        for (size_t i = 0; i < showCount; ++i) {
            if (i > 0) os << ", ";
            size_t idx = offset + i;
            const void* elemPtr = static_cast<const char*>(data) + idx * aclDataTypeBytes(desc->dtype);
            os << aclElementToString(desc->dtype, elemPtr);
        }
        offset += showCount;
        if (showCount < dimLen) {
            os << ", ...";
            truncated = true;
        }
        os << "]";
    } else {
        os << "[";
        size_t dimLen = dims[depth];
        size_t showCount = std::min(dimLen, maxShow - offset);
        if (showCount == 0) { os << "..."; return; }
        for (size_t i = 0; i < showCount; ++i) {
            if (i > 0) {
                os << ",\n" << std::string(baseIndent + depth + 1, ' ');
            }
            printTensorRecursive(os, desc, data, dims, depth + 1, offset, maxShow, truncated, baseIndent);
        }
        if (showCount < dimLen) {
            os << ",\n" << std::string(baseIndent + depth + 1, ' ') << "...";
            truncated = true;
        }
        os << "]";
    }
}

static std::string tensorDataToString(const aclTensorDesc* desc, const aclDataBuffer* buf) {
    if (!desc || !buf || !buf->data || buf->size == 0)
        return "(no data)";

    size_t numElements = calc_num_elements(desc, buf->size);
    if (numElements == 0) return "(no elements)";

    const std::vector<int64_t>& dims = desc->dims;
    const int baseIndent = 8;

    if (dims.empty()) {
        // Скаляр
        std::ostringstream oss;
        oss << std::string(baseIndent, ' ');
        oss << aclElementToString(desc->dtype, buf->data);
        return oss.str();
    }

    std::ostringstream oss;
    oss << std::string(baseIndent, ' ');
    size_t offset = 0;
    bool truncated = false;
    const size_t maxShow = 100;
    printTensorRecursive(oss, desc, buf->data, dims, 0, offset, maxShow, truncated, baseIndent);
    return oss.str();
}

static std::string formatTensorList(const char* label,
                                    const aclTensorDesc* const descs[],
                                    const aclDataBuffer* const bufs[],
                                    int count) {
    std::ostringstream oss;
    for (int i = 0; i < count; ++i) {
        oss << "    " << label << "[" << i << "]: ";
        if (descs[i]) {
            oss << tensorDescToString(descs[i]) << "\n";
            if (bufs[i] && bufs[i]->data && bufs[i]->size > 0) {
                oss << tensorDataToString(descs[i], bufs[i]) << "\n";
            } else {
                oss << "        (no buffer)\n";
            }
        } else {
            oss << "null\n";
        }
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
                               aclDataBuffer* const outputs[]);

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
                               aclDataBuffer* const outputs[]) { \
        BODY \
    } \
    static bool _reg_##NAME = (OpRegistry::add(#NAME, _op_##NAME), true)


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
        const int64_t* axes_ptr = static_cast<const int64_t*>(inputs[1]->data); \
        size_t num_axes = calc_num_elements(inputDesc[1], inputs[1]->size); \
        if (num_axes == 0) { \
            T acc = INIT; \
            for (size_t i = 0; i < in.numElements(); ++i) \
                acc = OP(acc, in[i]); \
            out[0] = acc; \
        } else { \
            int64_t axis = axes_ptr[0]; \
            if (axis < 0) axis += in.dims().size(); \
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
                T acc = INIT; \
                for (fullCoord[axis] = 0; fullCoord[axis] < in.dims()[axis]; ++fullCoord[axis]) { \
                    size_t inIdx = 0; \
                    size_t stride = 1; \
                    for (int i = in.dims().size()-1; i >= 0; --i) { \
                        inIdx += fullCoord[i] * stride; \
                        stride *= in.dims()[i]; \
                    } \
                    acc = OP(acc, in[inIdx]); \
                } \
                out[outIdx] = acc; \
            } \
        } \
        break; \
    }

#define DISPATCH_REDUCE_MIN(DT) \
    DISPATCH_REDUCE(DT, std::numeric_limits<T>::max(), [](T a, T b) { return a < b ? a : b; })

#define DISPATCH_REDUCE_MAX(DT) \
    DISPATCH_REDUCE(DT, std::numeric_limits<T>::lowest(), [](T a, T b) { return a > b ? a : b; })


#endif // HELPERS
