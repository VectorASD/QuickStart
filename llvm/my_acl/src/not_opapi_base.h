#include "common.h"
#include "op_profiler.h" // record_op_timing
#include "helpers.cpp"   // aclDataTypeToString, aclFormatToString, tensorDataToString

#include <cstdint>      // int32_t, int64_t, uint8_t
#include <ostream>
#include <sstream>


#ifndef NOT_OPAPI_BASE_H
#define NOT_OPAPI_BASE_H

struct aclOpExecutor {};

typedef int32_t aclnnStatus;
constexpr aclnnStatus OK = 0;
constexpr aclnnStatus UNIMPLEMENTED = 1;
constexpr aclnnStatus INVALID_PARAM = 2;
constexpr aclnnStatus UNASSERTED    = 3;
constexpr aclnnStatus MEMORY_FAULT  = 4;



#undef ASSERT_CODE
#undef ASSERT

struct AclnnException {
    aclnnStatus status;
    std::string message;
    AclnnException(aclnnStatus s, const std::string& msg) : status(s), message(msg) {}
};

struct OptionalScalarType {
    at::ScalarType value = at::ScalarType::Undefined;
    OptionalScalarType() = default;
    OptionalScalarType(at::ScalarType v) : value(v) {}
    operator bool() const { return value != at::ScalarType::Undefined; }
    operator at::ScalarType() const { return value; }
};

#define ASSERT_CODE(cond, code) \
    do { \
        if (!(cond)) { \
            std::ostringstream _log; \
            _log << "Error: " << __func__ << " assertion '" #cond "' failed"; \
            log_output(_log, true); \
            throw AclnnException((code), _log.str()); \
        } \
    } while (0);

#define ASSERT(cond) ASSERT_CODE(cond, UNASSERTED)

#define LOAD_TENSOR(tensor_var, acl_tensor_ptr, is_optional)                    \
    do {                                                                        \
        auto _tensor = (acl_tensor_ptr);                                        \
        if (!(acl_tensor_ptr) || !_tensor->desc || !_tensor->buffer) {          \
            if (is_optional) {                                                  \
                tensor_var = at::Tensor();                                      \
                break;                                                          \
            }                                                                   \
            throw AclnnException(UNIMPLEMENTED,                                 \
                                 "LOAD_TENSOR: null aclTensor or desc/buffer"); \
        }                                                                       \
        at::ScalarType type;                                                    \
        if (!try_toAtenType(_tensor->desc->dtype, type)) {                      \
            std::ostringstream log;                                             \
            log << "LOAD_TENSOR (" << acl_tensor_ptr                            \
                << ") unsupported data type '"                                  \
                << aclDataTypeToString(_tensor->desc->dtype) << '\'';           \
            throw AclnnException(UNIMPLEMENTED, log.str());                     \
        }                                                                       \
        auto opts = at::TensorOptions().dtype(type).device(at::kCPU);           \
        std::vector<int64_t> dims = _tensor->desc->dims;                        \
        std::vector<int64_t> strides = _tensor->strides;                        \
        int64_t offset = _tensor->offset;                                       \
        size_t elemSize = aclDataTypeBytes(_tensor->desc->dtype);               \
        tensor_var = at::from_blob(                                             \
            (char*)_tensor->buffer->data + offset * elemSize,                   \
            dims, strides, opts);                                               \
    } while (0)

#define DEFINE_ACLNN_OP(NAME, EXECUTOR_TYPE, ...)                \
    __attribute__((visibility("default")))                       \
    aclnnStatus NAME(void *workspace,                            \
                     uint64_t workspaceSize,                     \
                     aclOpExecutor *executor,                    \
                     aclrtStream stream) {                       \
        auto t_start = std::chrono::steady_clock::now();         \
        auto *exec = reinterpret_cast<EXECUTOR_TYPE*>(executor); \
        ASSERT(exec)                                             \
        std::ostringstream log;                                  \
        exec->start(#NAME, log);                                 \
        aclnnStatus _ret = OK;                                   \
        try {                                                    \
            ASSERT_CODE(!workspace && !workspaceSize && stream, INVALID_PARAM) \
            __VA_ARGS__                                          \
            exec->end(log);                                      \
        } catch (const AclnnException& e) {                      \
            _ret = e.status;                                     \
            log << "\n!!! " << e.message;                        \
        } catch (const std::exception& e) {                      \
            _ret = UNASSERTED;                                   \
            log << "\n!!! C++ exception: " << e.what();          \
        } catch (...) {                                          \
            _ret = UNASSERTED;                                   \
            log << "\n!!! unknown exception";                    \
        }                                                        \
        log_output(log, _ret != OK);                             \
        delete exec;                                             \
        auto t_end = std::chrono::steady_clock::now();           \
        double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count(); \
        record_op_timing(#NAME, elapsed_us);                     \
        return _ret;                                             \
    }



struct aclTensor {
    aclTensorDesc* desc;
    aclDataBuffer* buffer;
    std::vector<int64_t> strides;
    int64_t offset;
    bool already_sync = false;

    void store(const at::Tensor& src) {
        if (!desc)
            throw std::runtime_error("aclTensor::store: desc is null");

        size_t total_bytes = src.nbytes();

        // Если буфера нет или он слишком мал – выделяем новый
        if (!buffer || buffer->size < total_bytes) {
            void* new_data = malloc(total_bytes);
            if (!new_data) throw std::runtime_error("malloc failed in aclTensor::store");
            if (buffer) {
                if (buffer->owned)
                    free(buffer->data);
                DEBUG("CHECKKKKKKK: " << buffer->data)
                buffer->data = new_data;
                buffer->size = total_bytes;
                buffer->owned = true;
                DEBUG("  OK: " << buffer->data)
            } else
                buffer = new aclDataBuffer{new_data, total_bytes, true};
        }

        // Обновляем метаданные дескриптора (размеры, strides, offset)
        auto src_sizes = src.sizes();
        auto src_strides = src.strides();
        desc->dims.assign(src_sizes.begin(), src_sizes.end());
        strides.assign(src_strides.begin(), src_strides.end());
        offset = 0;

        // Копируем сырые данные
        std::memcpy(buffer->data, src.const_data_ptr(), total_bytes);

        // Устанавливаем флаг, чтобы SYNC_AFTER_MUTATION не делал лишнюю работу
        already_sync = true;
    }

    void sync_after_mutation(const at::Tensor& src) {
        if (already_sync) {
            already_sync = false; // сброс для будущих использований
            return;
        }
        auto sizes = src.sizes();
        auto strides = src.strides();
        desc->dims.assign(sizes.begin(), sizes.end());
        this->strides.assign(strides.begin(), strides.end());
        offset = 0;
    }
};

static inline std::ostream& tensorDataToString(const aclTensor* tensor, std::ostream& os, const int baseIndent = 8) {
    tensorDataToString(tensor->desc, tensor->buffer, tensor->strides, tensor->offset, os, baseIndent);
    auto& shape = tensor->desc->dims;
    auto& strides = tensor->strides;

    os << "   (";
    for (int i = 0; i < shape.size(); i++)
        os << (i ? ", " : "") << shape[i];
    os << ") " << aclDataTypeToString(tensor->desc->dtype) << " (";
    for (int i = 0; i < strides.size(); i++)
        os << (i ? ", " : "") << strides[i];
    return os << ')';
}

// тот случай, когда его невозможно сделать friend :)
inline std::ostream& operator<<(std::ostream& os, const aclTensor& tensor) {
    const auto& dims = tensor.desc->dims;
    bool small = dims.empty() ||
                 (dims.size() >= 1 &&
                  dims.back() <= 42 &&
                  std::all_of(dims.begin(), dims.end() - 1, [](int64_t d) { return d == 1; }));
    if (small)
        return tensorDataToString(&tensor, os, 0);
    os << '\n';
    return tensorDataToString(&tensor, os, 8);
}
inline std::ostream& operator<<(std::ostream& os, const aclTensor* tensor) {
    if (!tensor)
        return os << "(null)";
    return os << *tensor;
}



struct aclScalar {
    at::Scalar item;
    at::Tensor tensor;
    aclDataType dtype;

    friend std::ostream& operator<<(std::ostream& os, const aclScalar& scalar) {
        const void* ptr = scalar.tensor.const_data_ptr();
        return os << aclElementToString(scalar.dtype, ptr);
    }
    friend std::ostream& operator<<(std::ostream& os, const aclScalar* scalar) {
        if (scalar)
            return os << *scalar;
        return os << "null";
    }
};


struct aclIntArray {
    std::vector<int64_t> data;

    friend std::ostream& operator<<(std::ostream& os, const aclIntArray& arr) {
        os << '[';
        for (size_t i = 0; i < arr.data.size(); ++i)
            os << (i ? ", " : "") << arr.data[i];
        return os << ']';
    }
    friend std::ostream& operator<<(std::ostream& os, const aclIntArray* arr) {
        if (arr)
            return os << *arr;
        return os << "null";
    }
};

struct aclFloatArray {
    std::vector<float> data;

    friend std::ostream& operator<<(std::ostream& os, const aclFloatArray& arr) {
        os << '[';
        for (size_t i = 0; i < arr.data.size(); ++i)
            os << (i ? ", " : "") << formatFloat(arr.data[i]);
        return os << ']';
    }
    friend std::ostream& operator<<(std::ostream& os, const aclFloatArray* arr) {
        if (arr)
            return os << *arr;
        return os << "null";
    }
};

struct aclBoolArray {
    std::vector<uint8_t> data;  // 0|1

    friend std::ostream& operator<<(std::ostream& os, const aclBoolArray& arr) {
        os << '[';
        for (size_t i = 0; i < arr.data.size(); ++i)
            os << (i ? ", " : "") << (arr.data[i] ? "true" : "false");
        return os << ']';
    }
    friend std::ostream& operator<<(std::ostream& os, const aclBoolArray* arr) {
        if (arr)
            return os << *arr;
        return os << "null";
    }
};

struct aclScalarList {
    std::vector<const aclScalar*> scalars;

    friend std::ostream& operator<<(std::ostream& os, const aclScalarList& list) {
        os << "Scalar[";
        for (size_t i = 0; i < list.scalars.size(); ++i)
            os << (i ? ", " : "") << list.scalars[i];
        return os << ']';
    }
    friend std::ostream& operator<<(std::ostream& os, const aclScalarList* list) {
        if (list)
            return os << *list;
        return os << "null";
    }
};

static const std::vector<int64_t> _empty_int_vec;
static const std::vector<float> _empty_float_vec;
static const std::vector<uint8_t> _empty_bool_vec;
static const std::vector<const aclScalar*> _empty_scalar_list;


struct aclTensorList {
    std::vector<const aclTensor*> data; 
    int n;
    mutable std::vector<at::Tensor> cached; // кэш загруженных тензоров
    mutable bool loaded = false;

    aclTensorList(const aclTensor* const* ptrs, size_t count)
        : data(ptrs, ptrs + count), n(count) {}
            // Прямое хранение ptrs опасно для жизни программы (сразу отлетает segmentation fault при вызове toString)

    at::TensorList aten_tensors() const {
        if (!loaded) {
            cached.reserve(n);
            for (const aclTensor* t : data) {
                if (!t)
                    throw AclnnException(UNIMPLEMENTED, "null aclTensor in list");
                at::Tensor tensor;
                LOAD_TENSOR(tensor, t, 0);   // используем существующий макрос
                cached.push_back(tensor);
            }
            loaded = true;
        }
        return at::TensorList(cached);
    }

    static void toString(const char* name, const aclTensorList* list, std::ostringstream& oss) {
        oss << "\n    " << name << ": aclTensorList";
        if (!list) {
            oss << " null\n";
            return;
        }
        oss << '(' << list->n << "):";
        for (size_t i = 0; i < list->n; ++i) {
            oss << "\n        #" << i << ":\n";
            tensorDataToString(list->data[i], oss, 12);
        }
    }
};



// TODO: переделать cache в lru_cache (Least Recently Used - наименее недавно использованный)

#define CACHED_RANDOM_OP(OP_NAME, KEY_TYPE, KEY_MAKER, GEN_CODE, DTYPE_CODE) \
inline void cached_##OP_NAME##_(at::Tensor& self, int64_t seed, int64_t offset CACHED_##OP_NAME##_ARGS) { \
    size_t target_numel = self.numel(); \
    if (target_numel == 0) \
        return; \
    auto dtype = self.scalar_type(); \
    KEY_TYPE key = KEY_MAKER; \
    static thread_local auto gen = at::detail::getDefaultCPUGenerator(); \
    static thread_local std::map<KEY_TYPE, at::Tensor> cache; \
    static thread_local size_t default_cache_size = 65536; \
    auto it = cache.find(key); \
    if (it == cache.end()) { \
        gen.set_current_seed(static_cast<uint64_t>(seed)); \
        at::Tensor expander = at::empty({static_cast<int64_t>(default_cache_size)}, DTYPE_CODE); \
        { GEN_CODE } \
        it = cache.emplace(key, expander).first; \
    } \
    at::Tensor cache_tensor = it->second; \
    size_t cache_len = cache_tensor.numel(); \
    /* size_t start = static_cast<size_t>(offset) % cache_len; */ \
    size_t start = static_cast<size_t>((static_cast<uint64_t>(offset) * 0x9E3779B97F4A7C15ULL) >> 32) % target_numel; \
    /* убираем линейность start от медленно-растущего offset за счёт золотого сечения (φ)! \
       0x9E3779B97F4A7C15 = hex(floor(2^64 / φ)) = hex(math.floor(2**64 / примерно 1.618033988749895)) */ \
    size_t end = start + target_numel; \
    if (end > cache_len) { \
        std::ostringstream oss; \
        oss << "\n[EXPAND CACHE]"; \
        oss << "\nstart=" << start << ", end=" << end << ", cache_len=" << cache_len << ", target_numel=" << target_numel; \
        size_t new_size = target_numel * 2; \
        size_t expand_size = new_size - cache_len; \
        gen.set_current_seed(static_cast<uint64_t>(seed)); \
        at::Tensor expander = at::empty({static_cast<int64_t>(expand_size)}, DTYPE_CODE); \
        { GEN_CODE } \
        cache_tensor = it->second = at::cat({cache_tensor, expander}); \
        cache_len = new_size; \
        oss << "\nstart=" << start << ", end=" << end << ", cache_len=" << cache_len << ", target_numel=" << target_numel; \
        log_output(oss); \
    } \
    /* Копируем непосредственно в одномерный представление буфера */ \
    auto self_contiguous = at::from_blob( \
        self.data_ptr(), \
        {static_cast<int64_t>(target_numel)}, \
        {1}, \
        self.options() \
    ); \
    self_contiguous.copy_(cache_tensor.slice(0, start, end)); \
}


using NormalCacheKey   = std::tuple<int64_t, int64_t, float, float>;
using UniformCacheKey  = std::tuple<int64_t, int64_t, double, double>;
using RandomCacheKey   = std::tuple<int64_t, int64_t, int64_t, int64_t>;

#define CACHED_normal_ARGS , float mean, float std
CACHED_RANDOM_OP(
    normal,
    NormalCacheKey,
    NormalCacheKey(static_cast<int64_t>(dtype), seed, mean, std),
    expander.normal_(mean, std, gen);
    if (dtype != at::kFloat) expander = expander.to(dtype);, at::kFloat
)

#define CACHED_uniform_ARGS , double from, double to
CACHED_RANDOM_OP(
    uniform,
    UniformCacheKey,
    UniformCacheKey(static_cast<int64_t>(dtype), seed, from, to),
    expander.uniform_(from, to, gen);,
    dtype
)

#define CACHED_random_ARGS , int64_t from, int64_t to
CACHED_RANDOM_OP(
    random,
    RandomCacheKey,
    RandomCacheKey(static_cast<int64_t>(dtype), seed, from, to),
    expander.random_(from, to, gen);,
    dtype
)


#endif // NOT_OPAPI_BASE_H
