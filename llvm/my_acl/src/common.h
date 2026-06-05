#ifndef NOT_NPU_COMMON_H
#define NOT_NPU_COMMON_H

#include <cstddef>  // uint8_t, int16_t...
#include <cstdint>  // size_t
#include <iostream>       // std::cout
#include <string>         // std::string
#include <unordered_map>  // std::unordered_map
#include <vector>         // std::vector
#include <mutex>          // std::lock_guard, std::mutex
#include <sstream>        // std::ostringstream
#include <algorithm>      // std::sort

#if defined(_MSC_VER)
    #ifdef FUNC_VISIBILITY
        #define ACL_FUNC_VISIBILITY __declspec(dllexport)
    #else
        #define ACL_FUNC_VISIBILITY
    #endif
#else
    #ifdef FUNC_VISIBILITY
        #define ACL_FUNC_VISIBILITY __attribute__((visibility("default")))
    #else
        #define ACL_FUNC_VISIBILITY
    #endif
#endif

#if defined(__GNUC__) && (__GNUC__ >= 6)
    #define ACL_DEPRECATED __attribute__((deprecated))
    #define ACL_DEPRECATED_MESSAGE(message) __attribute__((deprecated(message)))
#elif defined(_MSC_VER)
    #define ACL_DEPRECATED __declspec(deprecated)
    #define ACL_DEPRECATED_MESSAGE(message) __declspec(deprecated(message))
#else
    #define ACL_DEPRECATED
    #define ACL_DEPRECATED_MESSAGE(message)
#endif

typedef enum {
    ACL_DEBUG = 0,
    ACL_INFO = 1,
    ACL_WARNING = 2,
    ACL_ERROR = 3,
} aclLogLevel;

typedef enum aclrtMemcpyKind {
    ACL_MEMCPY_HOST_TO_HOST,
    ACL_MEMCPY_HOST_TO_DEVICE,
    ACL_MEMCPY_DEVICE_TO_HOST,
    ACL_MEMCPY_DEVICE_TO_DEVICE,
    ACL_MEMCPY_DEFAULT,
    ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE,
    ACL_MEMCPY_INNER_DEVICE_TO_DEVICE,
    ACL_MEMCPY_INTER_DEVICE_TO_DEVICE,
} aclrtMemcpyKind;

typedef enum {
    ACL_DEVICE_INFO_UNDEFINED = -1,
    ACL_DEVICE_INFO_AI_CORE_NUM = 0,
    ACL_DEVICE_INFO_VECTOR_CORE_NUM = 1,
    ACL_DEVICE_INFO_L2_SIZE = 2
} aclDeviceInfo;

typedef enum aclrtMemAttr {
    ACL_DDR_MEM,
    ACL_HBM_MEM,
    ACL_DDR_MEM_HUGE,
    ACL_DDR_MEM_NORMAL,
    ACL_HBM_MEM_HUGE,
    ACL_HBM_MEM_NORMAL,
    ACL_DDR_MEM_P2P_HUGE,
    ACL_DDR_MEM_P2P_NORMAL,
    ACL_HBM_MEM_P2P_HUGE,
    ACL_HBM_MEM_P2P_NORMAL,
    ACL_HBM_MEM_HUGE1G,
    ACL_HBM_MEM_P2P_HUGE1G,
    ACL_MEM_NORMAL,
    ACL_MEM_HUGE,
    ACL_MEM_HUGE1G,
    ACL_MEM_P2P_NORMAL,
    ACL_MEM_P2P_HUGE,
    ACL_MEM_P2P_HUGE1G,
} aclrtMemAttr;

typedef void *aclrtStream;
typedef void *aclrtEvent;
typedef void *aclrtContext;
typedef void *aclrtNotify;
typedef void *aclrtCntNotify;
typedef void *aclrtLabel;
typedef void *aclrtLabelList;
typedef void *aclrtMbuf;
typedef int aclError;
typedef uint16_t aclFloat16;
typedef void *aclrtAllocatorDesc;
typedef void *aclrtAllocator;
typedef void *aclrtAllocatorBlock;
typedef void *aclrtAllocatorAddr;
typedef void *aclrtTaskGrp;

struct aclDataBuffer {
    void *data;
    size_t size;
};

static const int ACL_SUCCESS = 0;
static const int ACL_ERROR_INVALID_PARAM         = 100000;
static const int ACL_ERROR_OP_NOT_FOUND          = 100024;
static const int ACL_ERROR_UNSUPPORTED_DATA_TYPE = 100026;
static const int ACL_ERROR_BAD_ALLOC = 200000;

typedef enum {
    ACL_MEMTYPE_DEVICE = 0,
    ACL_MEMTYPE_HOST = 1,
    ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT = 2
} aclMemType;

static const char* aclMemTypeToString(aclMemType memType) {
    switch (memType) {
        case ACL_MEMTYPE_DEVICE: return "device";
        case ACL_MEMTYPE_HOST:   return "host";
        case ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT: return "host_compile_indep";
        default:                 return "unknown";
    }
}

typedef enum {
    ACL_FORMAT_UNDEFINED = -1,
    ACL_FORMAT_NCHW = 0,
    ACL_FORMAT_NHWC = 1,
    ACL_FORMAT_ND = 2,
    ACL_FORMAT_NC1HWC0 = 3,
    ACL_FORMAT_FRACTAL_Z = 4,
    ACL_FORMAT_NC1HWC0_C04 = 12,
    ACL_FORMAT_HWCN = 16,
    ACL_FORMAT_NDHWC = 27,
    ACL_FORMAT_FRACTAL_NZ = 29,
    ACL_FORMAT_NCDHW = 30,
    ACL_FORMAT_NDC1HWC0 = 32,
    ACL_FRACTAL_Z_3D = 33,
    ACL_FORMAT_NC = 35,
    ACL_FORMAT_NCL = 47,
    ACL_FORMAT_FRACTAL_NZ_C0_16 = 50,
    ACL_FORMAT_FRACTAL_NZ_C0_32 = 51,
    ACL_FORMAT_FRACTAL_NZ_C0_2 = 52,
    ACL_FORMAT_FRACTAL_NZ_C0_4 = 53,
    ACL_FORMAT_FRACTAL_NZ_C0_8 = 54,
} aclFormat;

typedef enum {
    ACL_DT_UNDEFINED = -1,
    ACL_FLOAT = 0,
    ACL_FLOAT16 = 1,
    ACL_INT8 = 2,
    ACL_INT32 = 3,
    ACL_UINT8 = 4,
    ACL_INT16 = 6,
    ACL_UINT16 = 7,
    ACL_UINT32 = 8,
    ACL_INT64 = 9,
    ACL_UINT64 = 10,
    ACL_DOUBLE = 11,
    ACL_BOOL = 12,
    ACL_STRING = 13,
    ACL_COMPLEX64 = 16,
    ACL_COMPLEX128 = 17,
    ACL_BF16 = 27,
    ACL_INT4 = 29,
    ACL_UINT1 = 30,
    ACL_COMPLEX32 = 33,
    ACL_HIFLOAT8 = 34,
    ACL_FLOAT8_E5M2 = 35,
    ACL_FLOAT8_E4M3FN = 36,
    ACL_FLOAT8_E8M0 = 37,
    ACL_FLOAT6_E3M2 = 38,
    ACL_FLOAT6_E2M3 = 39,
    ACL_FLOAT4_E2M1 = 40,
    ACL_FLOAT4_E1M2 = 41,
    ACL_HIFLOAT4 = 42,
} aclDataType;

static size_t aclDataTypeBits(aclDataType dtype) {
    switch (dtype) {
        case ACL_FLOAT:           return 32;
        case ACL_FLOAT16:         return 16;
        case ACL_INT8:            return 8;
        case ACL_INT32:           return 32;
        case ACL_UINT8:           return 8;
        case ACL_INT16:           return 16;
        case ACL_UINT16:          return 16;
        case ACL_UINT32:          return 32;
        case ACL_INT64:           return 64;
        case ACL_UINT64:          return 64;
        case ACL_DOUBLE:          return 64;
        case ACL_BOOL:            return 1;
        case ACL_STRING:          return 0;  // неизвестен
        case ACL_COMPLEX64:       return 64; // два float
        case ACL_COMPLEX128:      return 128; // два double
        case ACL_BF16:            return 16;
        case ACL_INT4:            return 4;
        case ACL_UINT1:           return 1;
        case ACL_COMPLEX32:       return 64; // условно
        case ACL_HIFLOAT8:        return 8;
        case ACL_FLOAT8_E5M2:     return 8;
        case ACL_FLOAT8_E4M3FN:   return 8;
        case ACL_FLOAT8_E8M0:     return 8;
        case ACL_FLOAT6_E3M2:     return 6;
        case ACL_FLOAT6_E2M3:     return 6;
        case ACL_FLOAT4_E2M1:     return 4;
        case ACL_FLOAT4_E1M2:     return 4;
        case ACL_HIFLOAT4:        return 4;
        default:                  return 8;  // неизвестный тип – считаем байт
    }
}

static size_t aclDataTypeBytes(aclDataType dtype) {
    return (aclDataTypeBits(dtype) + 7) / 8;
}

struct aclTensorDesc {
    aclDataType dtype;
    aclFormat format;
    std::vector<int64_t> dims;
    std::string name;
    aclMemType memType;

    aclTensorDesc()
        : dtype(ACL_DT_UNDEFINED),
          format(ACL_FORMAT_UNDEFINED),
          memType(ACL_MEMTYPE_DEVICE) {}
};

#define ACL_UNKNOWN_RANK 0xFFFFFFFFFFFFFFFE

struct aclopAttr {
    std::unordered_map<std::string, uint8_t> bools;
    std::unordered_map<std::string, int64_t> ints;
    std::unordered_map<std::string, float> floats;
    std::unordered_map<std::string, std::string> strings;
    std::unordered_map<std::string, aclDataType> dtypes;

    std::unordered_map<std::string, std::vector<uint8_t>> list_bools;
    std::unordered_map<std::string, std::vector<int64_t>> list_ints;
    std::unordered_map<std::string, std::vector<float>> list_floats;
    std::unordered_map<std::string, std::vector<std::vector<int64_t>>> list_list_ints;
};


// безопасное логирование

static std::mutex g_log_mutex;

static bool log_is_quiet() {
    static int quiet = -1;
    if (quiet < 0) {
        const char* env = std::getenv("NOT_NPU_QUIET");
        quiet = (env && env[0] == '1') ? 1 : 0;
    }
    return quiet == 1;
}

static inline void log_output(const std::ostringstream& oss, bool is_error = false) {
    static bool can_log = !log_is_quiet();
    if (is_error || can_log) {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        std::cout << oss.str() << std::endl;
    }
}
static inline void log_output(const std::string& msg, bool is_error = false) {
    static bool can_log = !log_is_quiet();
    if (is_error || can_log) {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        std::cout << msg << std::endl;
    }
}


#endif // NOT_NPU_COMMON_H
