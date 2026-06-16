#include "op_profiler.h" // record_op_timing
#include "helpers.cpp"   // aclDataTypeToString, aclFormatToString, tensorDataToString

#include <cstdint>      // int32_t, int64_t, uint64_t


#ifndef NOT_OPAPI_BASE_H
#define NOT_OPAPI_BASE_H

struct aclOpExecutor {};
struct aclIntArray {};
struct aclFloatArray {};
struct aclBoolArray {};
struct aclScalarList {};

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

#define LOAD_TENSOR(tensor_var, acl_tensor_ptr)                                 \
    do {                                                                        \
        auto _tensor = (acl_tensor_ptr);                                        \
        if (!(acl_tensor_ptr) || !_tensor->desc || !_tensor->buffer) {          \
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
        log_output(log, true);                                   \
        delete exec;                                             \
        auto t_end = std::chrono::steady_clock::now();           \
        double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count(); \
        record_op_timing(#NAME, elapsed_us);                     \
        return _ret;                                             \
    }

#define SYNC_AFTER_MUTATION(acl_tensor_ptr, at_tensor) \
    do { \
        auto _t = (acl_tensor_ptr);                                     \
        auto& _at = (at_tensor);                                        \
        _t->desc->dims.assign(_at.sizes().begin(), _at.sizes().end());  \
        _t->strides.assign(_at.strides().begin(), _at.strides().end()); \
        /* offset остаётся 0, т.к. после операции тензор непрерывен */  \
        _t->offset = 0;                                                 \
    } while (0);



struct aclTensor {
    aclTensorDesc* desc;
    aclDataBuffer* buffer;
    std::vector<int64_t> strides;
    int64_t offset;
};

static inline std::string tensorDataToString(const aclTensor* tensor) {
    return tensorDataToString(tensor->desc, tensor->buffer, tensor->strides, tensor->offset);
}



struct aclScalar {
    at::Tensor tensor;
    aclDataType dtype;

    std::string toString() const {
        const void* ptr = tensor.const_data_ptr();
        return aclElementToString(dtype, ptr);
    }
};



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
                LOAD_TENSOR(tensor, t);   // используем существующий макрос
                cached.push_back(tensor);
            }
            loaded = true;
        }
        return at::TensorList(cached);
    }

    static void toString(const aclTensorList* list, std::ostringstream& oss) {
        if (!list) {
            oss << "\n    aclTensorList null\n";
            return;
        }
        oss << "\n    aclTensorList(" << list->n << "):";
        for (size_t i = 0; i < list->n; ++i)
            oss << "\n      [" << i << "]:\n" << tensorDataToString(list->data[i]);
    }
};


#endif // NOT_OPAPI_BASE_H
