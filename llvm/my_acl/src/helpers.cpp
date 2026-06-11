#include "not_acl.cpp"  // aclGetTensorDescDimV2, aclGetTensorDescNumDims, aclGetTensorDescType
#include <algorithm>    // copy, sort

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

// float16: знак 1 бит, экспонента 5, мантисса 10
// float32: знак 1 бит, экспонента 8, мантисса 23
inline float half_to_float(uint16_t v) {
    at::Tensor t = at::from_blob(&v, {1}, at::TensorOptions().dtype(at::kHalf));
    return t.item<float>();
}
inline uint16_t float_to_half(float f) {
    at::Tensor t = at::from_blob(&f, {1}, at::TensorOptions().dtype(at::kFloat));
    return t.to(at::kHalf).item<c10::Half>().x;
}

// float16: знак 1 бит, экспонента 8, мантисса 7
// float32: знак 1 бит, экспонента 8, мантисса 23
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

template <typename T>
static std::string formatFloatValue(T val) {
    std::ostringstream oss;
    if constexpr (std::is_same_v<T, float>) {
        if (std::abs(val) >= 1e-4f && std::abs(val) < 1e4f)
            oss << std::fixed << std::setprecision(4) << val;
        else
            oss << std::scientific << std::setprecision(4) << val;
    } else {
        if (std::abs(val) >= 1e-4 && std::abs(val) < 1e4)
            oss << std::fixed << std::setprecision(4) << val;
        else
            oss << std::scientific << std::setprecision(4) << val;
    }
    return oss.str();
}

// Преобразует один элемент тензора в строку по его типу
static std::string aclElementToString(aclDataType dtype, const void* elemPtr) {
    if (!elemPtr) return "?";

    auto formatFloat = [](float val) { return formatFloatValue(val); };
    auto formatDouble = [](double val) { return formatFloatValue(val); };

    switch (dtype) {
        case ACL_FLOAT:   return formatFloat(*static_cast<const float*>(elemPtr));
        case ACL_DOUBLE:  return formatDouble(*static_cast<const double*>(elemPtr));
        case ACL_FLOAT16: return formatFloat(half_to_float(*static_cast<const uint16_t*>(elemPtr)));
        case ACL_BF16:    return formatFloat(bf16_to_float(*static_cast<const uint16_t*>(elemPtr)));
        case ACL_INT8:    return std::to_string(static_cast<int>(*static_cast<const int8_t*>(elemPtr)));
        case ACL_UINT8:   return std::to_string(static_cast<unsigned>(*static_cast<const uint8_t*>(elemPtr)));
        case ACL_INT16:   return std::to_string(*static_cast<const int16_t*>(elemPtr));
        case ACL_UINT16:  return std::to_string(*static_cast<const uint16_t*>(elemPtr));
        case ACL_INT32:   return std::to_string(*static_cast<const int32_t*>(elemPtr));
        case ACL_UINT32:  return std::to_string(*static_cast<const uint32_t*>(elemPtr));
        case ACL_INT64:   return std::to_string(*static_cast<const int64_t*>(elemPtr));
        case ACL_UINT64:  return std::to_string(*static_cast<const uint64_t*>(elemPtr));
        case ACL_BOOL:    return *static_cast<const bool*>(elemPtr) ? "true" : "false";

        case ACL_COMPLEX32: {
            const uint16_t* p = static_cast<const uint16_t*>(elemPtr);
            float real = half_to_float(p[0]);
            float imag = half_to_float(p[1]);
            if (real == 0.0f && imag == 0.0f)
                return "0";
            std::ostringstream oss;
            oss << formatFloat(real)
                << (imag >= 0 && !std::signbit(imag) ? "+" : "")
                << formatFloat(imag) << "j";
            return oss.str();
        }
        case ACL_COMPLEX64: {
            const float* p = static_cast<const float*>(elemPtr);
            if (p[0] == 0.0f && p[1] == 0.0f)
                return "0";
            std::ostringstream oss;
            oss << formatFloat(p[0])
                << (p[1] >= 0 && !std::signbit(p[1]) ? "+" : "")
                << formatFloat(p[1]) << "j";
            return oss.str();
        }
        case ACL_COMPLEX128: {
            const double* p = static_cast<const double*>(elemPtr);
            if (p[0] == 0.0 && p[1] == 0.0)
                return "0";
            std::ostringstream oss;
            oss << formatDouble(p[0])
                << (p[1] >= 0 && !std::signbit(p[1]) ? "+" : "")
                << formatDouble(p[1]) << "j";
            return oss.str();
        }
        default:
            return "?";
    }
}

static std::string tensorDataToString(const aclTensorDesc* desc, const aclDataBuffer* buf, const std::vector<int64_t>& strides, int64_t offset) {
    if (!desc || !buf || !buf->data || buf->size == 0)
        return "(no data)";

    size_t numElements = calc_num_elements(desc, buf->size);
    if (numElements == 0) return "(no elements)";

    const std::vector<int64_t>& dims = desc->dims;
    const int baseIndent = 8;
    const int edgeItems = 3;          // сколько элементов показывать с краёв при сокращении
    const size_t threshold = 1000;    // порог общего числа элементов для включения сокращения
    bool doTruncate = numElements > threshold;

    const char* baseData = static_cast<const char*>(buf->data);
    size_t elemSize = aclDataTypeBytes(desc->dtype);

    std::ostringstream oss;
    oss << std::string(baseIndent, ' ');
    if (dims.empty()) {
        const char* ptr = baseData + offset * aclDataTypeBytes(desc->dtype);
        oss << aclElementToString(desc->dtype, ptr);
        return oss.str();
    }

    // Рекурсивная лямбда для форматирования
    std::function<void(std::ostringstream&, const std::vector<int64_t>&, int, int, int64_t)> printRec;
    printRec = [&](std::ostringstream& oss, const std::vector<int64_t>& curDims, int depth, int indent, int64_t curOffset) {
        if (depth == curDims.size() - 1) {
            // Последнее измерение – строка чисел
            int64_t size = curDims[depth];
            oss << '[';
            if (doTruncate && size > 2 * edgeItems + 1) {
                for (int64_t i = 0; i < edgeItems; ++i) {
                    if (i > 0) oss << ", ";
                    const void* elemPtr = baseData + (curOffset + i * strides[depth]) * elemSize;
                    oss << aclElementToString(desc->dtype, elemPtr);
                }
                oss << ", ..., ";
                for (int64_t i = size - edgeItems; i < size; ++i) {
                    if (i > size - edgeItems) oss << ", ";
                    const void* elemPtr = baseData + (curOffset + i * strides[depth]) * elemSize;
                    oss << aclElementToString(desc->dtype, elemPtr);
                }
            } else {
                for (int64_t i = 0; i < size; ++i) {
                    if (i > 0) oss << ", ";
                    const void* elemPtr = baseData + (curOffset + i * strides[depth]) * elemSize;
                    oss << aclElementToString(desc->dtype, elemPtr);
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
                for (int64_t i = 0; i < edgeItems; ++i) {
                    if (i > 0) oss << ",\n" << pad;
                    int64_t newOffset = curOffset + i * strides[depth];
                    printRec(oss, curDims, depth + 1, indent, newOffset);
                }
                oss << ",\n" << std::string(indent + 1, ' ')
                    << "...\n" << std::string(indent, ' ');
                for (int64_t i = size - edgeItems; i < size; ++i) {
                    if (i > size - edgeItems) oss << ",\n" << pad;
                    int64_t newOffset = curOffset + i * strides[depth];
                    printRec(oss, curDims, depth + 1, indent, newOffset);
                }
            } else {
                for (int64_t i = 0; i < size; ++i) {
                    if (i > 0) oss << ",\n" << pad;
                    int64_t newOffset = curOffset + i * strides[depth];
                    printRec(oss, curDims, depth + 1, indent, newOffset);
                }
            }
            oss << ']';
        }
    };

    printRec(oss, dims, 0, baseIndent, offset);
    return oss.str();
}

static std::string tensorDataToString(const aclTensorDesc* desc, const aclDataBuffer* buf) {
    std::vector<int64_t> denseStrides(desc->dims.size());
    if (!desc->dims.empty()) {
        denseStrides.back() = 1;
        for (int i = desc->dims.size()-2; i >= 0; --i)
            denseStrides[i] = denseStrides[i+1] * desc->dims[i+1];
    }
    return tensorDataToString(desc, buf, denseStrides, 0);
}

enum TensorPrintFlags {
    PRINT_DESC = 1 << 0,
    PRINT_DATA = 1 << 1,
    PRINT_ALL  = PRINT_DESC | PRINT_DATA
};

static void formatTensorList(std::ostringstream &oss,
                                    const char* label,
                                    const aclTensorDesc* const descs[],
                                    const aclDataBuffer* const bufs[],
                                    int count,
                                    int flags = PRINT_ALL) {
    for (int i = 0; i < count; ++i) {
        if (descs[i]) {
            if (flags & PRINT_DESC) {
                oss << "\n    " << label << "[" << i << "]: ";
                tensorDescToString(descs[i], oss);
            }
            if (flags & PRINT_DATA) {
                if (bufs[i] && bufs[i]->data)
                    oss << '\n' << tensorDataToString(descs[i], bufs[i]);
                else
                    oss << "\n    (no buffer)";
            }
        } else
            oss << "\nnull";
    }
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
                               const aclopAttr* attrs);

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

#define REGISTER_OP(NAME, ...) \
    static exitCode _op_##NAME(int numInputs, const aclTensorDesc* const inputDesc[], \
                               const aclDataBuffer* const inputs[], \
                               int numOutputs, const aclTensorDesc* const outputDesc[], \
                               aclDataBuffer* const outputs[], \
                               const aclopAttr* attr) { \
        __VA_ARGS__ \
    } \
    static bool _reg_##NAME = (OpRegistry::add(#NAME, _op_##NAME), true)
// За счёт __VA_ARGS__, теперь можно писать запятые прямо в BODY (что заменён на '...')

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
        case ACL_COMPLEX32:  return at::kComplexHalf;
        case ACL_COMPLEX64:  return at::kComplexFloat;
        case ACL_COMPLEX128: return at::kComplexDouble;
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
    auto it = attr->values.find(key);
    if (it == attr->values.end())
        return false;

    if constexpr (std::is_same_v<T, int>) {
        const int64_t* p = std::get_if<int64_t>(&it->second);
        if (p) {
            value = static_cast<int>(*p);
            return true;
        }
        return false;
    } else if constexpr (std::is_same_v<T, bool>) {
        const uint8_t* p = std::get_if<uint8_t>(&it->second);
        if (p) {
            value = (*p != 0);
            return true;
        }
        return false;
    } else {
        static_assert(
            std::disjunction_v<
                std::is_same<T, int64_t>,
                std::is_same<T, float>,
                std::is_same<T, std::string>,
                std::is_same<T, aclDataType>,
                std::is_same<T, std::vector<int64_t>>,
                std::is_same<T, std::vector<uint8_t>>,
                std::is_same<T, std::vector<float>>,
                std::is_same<T, std::vector<std::vector<int64_t>>>
            >,
            "Unsupported attribute type"
        );
        const T* p = std::get_if<T>(&it->second);
        if (p) {
            value = *p;
            return true;
        }
        return false;
    }
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
        int64_t axis; \
        ASSERT(try_get_attr<int64_t>(attr, "dimension", axis)) \
        if (axis < 0) axis += in.dims().size(); \
        \
        bool keepdim = false; \
        try_get_attr<bool>(attr, "keep_dims", keepdim); /* необязательный атрибут, по умолчанию false */ \
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
        int64_t axis; \
        ASSERT(try_get_attr<int64_t>(attr, "dimension", axis)) \
        if (axis < 0) axis += in.dims().size(); \
        \
        bool keepdim = false; \
        try_get_attr<bool>(attr, "keep_dims", keepdim); /* необязательный атрибут, по умолчанию false */ \
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
        std::vector<int64_t> axes; \
        ASSERT(try_get_attr<std::vector<int64_t>>(attr, "axes", axes)) \
        \
        /* keep_dims (опционально) */ \
        bool keep_dims = false; \
        try_get_attr<bool>(attr, "keep_dims", keep_dims); /* необязательный атрибут, по умолчанию false */ \
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
