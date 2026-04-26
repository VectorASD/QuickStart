#!/bin/bash
# Данный сборщик библиотек призван полностью заменить нестабильные not_torch.py и not_aten.py.
# Низкоуровневый API хотя бы имеет конечный объём работы, в отличие от экспоненциально растущих
#   Python‑патчей, пытающихся одновременно угодить всем слоям PyTorch.

set -e

: << 'COMMENT'
cd ~/tmp
cp /mnt/c/Users/VectorASD/Downloads/Ascend-cann_9.0.0-beta.2_linux-x86_64.run .
./Ascend-cann_9.0.0-beta.2_linux-x86_64.run --noexec --extract=Ascend-cann
rm Ascend-cann_9.0.0-beta.2_linux-x86_64.run

cd ~/tmp/Ascend-cann/run_package
for f in *.run; do ./"$f" --noexec --extract="${f%%_*}"; done
rm *.run

find . -name libhccl.so
find . -name libhccl*.so
    ./cann-hcomm/hcomm/lib64/libhccl_v2_native_alg_repo.so
    ./cann-hcomm/hcomm/lib64/libhccl_v2.so
    ./cann-hcomm/hcomm/lib64/libhccl_plf.so
    ./cann-hcomm/hcomm/lib64/libhccl_v2_alg_frame.so
    ./cann-hcomm/hcomm/lib64/libhccl_alg.so
    ./cann-hcomm/hcomm/lib64/libhccl_legacy.so
find . -name libascendcl.so
    ./cann-npu-runtime/runtime/lib/libascendcl.so
    ./cann-npu-runtime/runtime/lib/stub/linux/x86_64/libascendcl.so
    ./cann-npu-runtime/runtime/devlib/linux/aarch64/libascendcl.so
    ./cann-npu-runtime/runtime/devlib/linux/x86_64/libascendcl.so
find . -name libacl_op_compiler.so
    ./cann-ge-compiler/ge-compiler/lib64/libacl_op_compiler.so
    ./cann-ge-compiler/ge-compiler/lib64/stub/linux/aarch64/libacl_op_compiler.so
    ./cann-ge-compiler/ge-compiler/lib64/stub/linux/x86_64/libacl_op_compiler.so
find . -name libge_runner.so
    ./cann-ge-compiler/ge-compiler/lib64/libge_runner.so
    ./cann-ge-compiler/ge-compiler/lib64/stub/linux/aarch64/libge_runner.so
    ./cann-ge-compiler/ge-compiler/lib64/stub/linux/x86_64/libge_runner.so
find . -name libgraph.so
    ./cann-ge-executor/ge-executor/lib64/libgraph.so
    ./cann-ge-executor/ge-executor/lib64/stub/linux/aarch64/libgraph.so
    ./cann-ge-executor/ge-executor/lib64/stub/linux/x86_64/libgraph.so
    ./cann-ge-executor/ge-executor/lib64/stub/minios/aarch64/libgraph.so
find . -name libacl_tdt_channel.so
    ./cann-npu-runtime/runtime/lib/libacl_tdt_channel.so
    ./cann-npu-runtime/runtime/devlib/linux/aarch64/libacl_tdt_channel.so
    ./cann-npu-runtime/runtime/devlib/linux/x86_64/libacl_tdt_channel.so
COMMENT


mkdir -p lib



if [ ! -f lib/libhccl.so ]; then
    cat > lib/not_hccl.c << 'EOF'
void __not_hccl_placeholder() {}
EOF

    gcc -shared -fPIC lib/not_hccl.c -o lib/libhccl.so

    echo "Created lib/libhccl.so"
fi



common=$(cat << 'EOF'
#include <thread>        // std::thread
#include <unordered_map> // std::unordered_map
#include <vector>        // std::vector


#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define ACL_FUNC_VISIBILITY _declspec(dllexport)
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

static const int ACL_ERROR_INVALID_PARAM = 100000;
static const int ACL_ERROR_BAD_ALLOC = 200000;


typedef enum {
  ACL_MEMTYPE_DEVICE = 0,
  ACL_MEMTYPE_HOST = 1,
  ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT = 2
} aclMemType;

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


static inline void log_ptr(const char* name, const void* ptr) {
    std::cout << name << "=0x"
              << std::hex << reinterpret_cast<size_t>(ptr)
              << std::dec;
}
EOF
)



cat > lib/not_acl.cpp << EOF
#include <iostream>
#include <cstdarg>       // va_start, va_end
#include <cstring>       // memcpy
$common

void __not_acl_placeholder() {}

static uint32_t g_device_count = 1;
static uint32_t g_current_device = 0;

static int64_t info_ai_core_num     = 28;
static int64_t info_cube_core_num   = info_ai_core_num;
static int64_t info_vector_core_num = info_ai_core_num * 2;
static int64_t info_L2_size         = 112LL * 1024 * 1024;               // 112 MiB
static int64_t info_GM_size         = (79LL * 1024 + 736) * 1024 * 1024; // 79 Gb, 736 Mb


#ifdef __cplusplus
extern "C" {
#endif


// ~~~ cann-npu-runtime/runtime/include/external/acl/acl_base_rt.h ~~~

ACL_FUNC_VISIBILITY void aclAppLog(aclLogLevel logLevel, const char *func, const char *file, uint32_t line,
                                   const char *fmt, ...) {
    std::cout << "[aclAppLog] level=" << logLevel
              << " func=" << (func ? func : "<null>")
              << " file=" << (file ? file : "<null>")
              << " line=" << line << std::endl;

    char buffer[8192];

    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    std::cout << "    msg: " << buffer << std::endl;
}

ACL_FUNC_VISIBILITY aclError aclGetDeviceCapability(uint32_t deviceId, aclDeviceInfo deviceInfo, int64_t *value) {
    std::cout << "[aclGetDeviceCapability] deviceId=" << deviceId << " deviceInfo=" << deviceInfo << " ";
    log_ptr("value_ptr", value);
    std::cout << std::endl;

    if (!value) {
        std::cout << "    value is null → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    switch (deviceInfo) {
        case ACL_DEVICE_INFO_AI_CORE_NUM:
            *value = info_ai_core_num;
            break;
        case ACL_DEVICE_INFO_VECTOR_CORE_NUM:
            *value = info_vector_core_num;
            break;
        case ACL_DEVICE_INFO_L2_SIZE:
            *value = info_L2_size;
            break;
        default:
            std::cout << "    unknown deviceInfo → ACL_ERROR_INVALID_PARAM" << std::endl;
            return ACL_ERROR_INVALID_PARAM;
    }

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclDataBuffer *aclCreateDataBuffer(void *data, size_t size) {
    std::cout << "[aclCreateDataBuffer] ";
    log_ptr("data", data);
    std::cout << " size=" << size << std::endl;

    aclDataBuffer *buf = new aclDataBuffer();
    if (!buf) {
        std::cout << "    new failed → nullptr" << std::endl;
        return nullptr;
    }

    buf->data = data;
    buf->size = size;

    std::cout << "    created aclDataBuffer ";
    log_ptr("ptr", buf);
    std::cout << std::endl;

    return buf;
}

ACL_FUNC_VISIBILITY aclError aclDestroyDataBuffer(const aclDataBuffer *dataBuffer) {
    std::cout << "[aclDestroyDataBuffer] ";
    log_ptr("dataBuffer", dataBuffer);
    std::cout << std::endl;

    delete dataBuffer;
    return ACL_SUCCESS;
}



// ~~~ cann-npu-runtime/runtime/include/external/acl/acl_rt.h ~~~

typedef enum {
    ACL_DEV_ATTR_AICPU_CORE_NUM  = 1,    // number of AI CPUs

    ACL_DEV_ATTR_AICORE_CORE_NUM = 101,  // number of AI Cores
    ACL_DEV_ATTR_CUBE_CORE_NUM   = 102,  // number of Cube Cores

    ACL_DEV_ATTR_VECTOR_CORE_NUM = 201,  // number of Vector Cores
    ACL_DEV_ATTR_WARP_SIZE       = 202,  // number of threads in a Warp
    ACL_DEV_ATTR_MAX_THREAD_PER_VECTOR_CORE = 203,    // maximum number of concurrent threads per Vector Core
    ACL_DEV_ATTR_LOCAL_MEM_PER_VECTOR_CORE
 	    ACL_DEPRECATED_MESSAGE("Use ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE instead") = 204,    // DEPRECATED: Use ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE
    ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE  = 204,    // maximum available local memory per Vector Core, in Bytes

    ACL_DEV_ATTR_TOTAL_GLOBAL_MEM_SIZE = 301,    // total available global memory on the Device, in Bytes
    ACL_DEV_ATTR_L2_CACHE_SIZE         = 302,    // L2 Cache size, in Bytes

    ACL_DEV_ATTR_SMP_ID = 401U,                 // indicates whether devices are on the same OS
    ACL_DEV_ATTR_PHY_CHIP_ID = 402U,            // physical chip id
    ACL_DEV_ATTR_SUPER_POD_DEVIDE_ID = 403U,    // DEPRECATED: Use ACL_DEV_ATTR_SUPER_POD_DEVICE_ID
    ACL_DEV_ATTR_SUPER_POD_DEVICE_ID = 403U,    // super pod device id
    ACL_DEV_ATTR_SUPER_POD_SERVER_ID = 404U,    // super pod server id
    ACL_DEV_ATTR_SUPER_POD_ID = 405U,           // super pod id
    ACL_DEV_ATTR_CUST_OP_PRIVILEGE = 406U,      // indicates whether the custom operator privilege is enabled
    ACL_DEV_ATTR_MAINBOARD_ID = 407U,           // mainborad id

    ACL_DEV_ATTR_IS_VIRTUAL = 501U,             // whether it is in compute power splitting mode
} aclrtDevAttr;


ACL_FUNC_VISIBILITY aclError aclrtSynchronizeStream(aclrtStream stream) {
    std::cout << "[aclrtSynchronizeStream] ";
    log_ptr("stream", stream);
    std::cout << std::endl;

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtGetDeviceCount(uint32_t *count) {
    *count = g_device_count;
    std::cout << "[aclrtGetDeviceCount] count=" << *count << std::endl;

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY const char *aclGetRecentErrMsg() {
    std::cout << "[aclGetRecentErrMsg]" << std::endl;
    return "not-npu: no recent error";
}

ACL_FUNC_VISIBILITY aclError aclrtFreeHost(void *hostPtr) {
    std::cout << "[aclrtFreeHost] ";
    log_ptr("hostPtr", hostPtr);
    std::cout << std::endl;

    free(hostPtr);

    return ACL_SUCCESS;
}

typedef enum aclrtMemMallocPolicy {
    ACL_MEM_MALLOC_HUGE_FIRST,
    ACL_MEM_MALLOC_HUGE_ONLY,
    ACL_MEM_MALLOC_NORMAL_ONLY,
    ACL_MEM_MALLOC_HUGE_FIRST_P2P,
    ACL_MEM_MALLOC_HUGE_ONLY_P2P,
    ACL_MEM_MALLOC_NORMAL_ONLY_P2P,
    ACL_MEM_MALLOC_HUGE1G_ONLY,
    ACL_MEM_MALLOC_HUGE1G_ONLY_P2P,
    ACL_MEM_TYPE_LOW_BAND_WIDTH   = 0x0100,
    ACL_MEM_TYPE_HIGH_BAND_WIDTH  = 0x1000,
    ACL_MEM_ACCESS_USER_SPACE_READONLY = 0x100000,
} aclrtMemMallocPolicy;

ACL_FUNC_VISIBILITY aclError aclrtMalloc(void **devPtr,
                                         size_t size,
                                         aclrtMemMallocPolicy policy) {
    std::cout << "[aclrtMalloc] size=" << size
              << " policy=" << policy << " ";
    log_ptr("devPtr", devPtr);
    std::cout << std::endl;

    if (!devPtr) {
        std::cout << "    devPtr is null → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    // not‑NPU: device memory = обычный malloc
    void *ptr = malloc(size);
    *devPtr = ptr;

    std::cout << "    ";
    log_ptr("allocated", ptr);
    std::cout << std::endl;

    if (!ptr) {
        std::cout << "    allocation failed → ACL_ERROR_BAD_ALLOC" << std::endl;
        return ACL_ERROR_BAD_ALLOC;
    }

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtMallocHost(void **hostPtr, size_t size) {
    std::cout << "[aclrtMallocHost] size=" << size << std::endl;

    void *ptr = malloc(size);
    *hostPtr = ptr;

    std::cout << "    ";
    log_ptr("allocated", ptr);
    std::cout << std::endl;

    if (!ptr) {
        std::cout << "    allocation failed → ACL_ERROR_BAD_ALLOC" << std::endl;
        return ACL_ERROR_BAD_ALLOC;
    }

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtGetDevice(int32_t *deviceId) {
    std::cout << "[aclrtGetDevice] ";
    log_ptr("deviceId_ptr", deviceId);
    std::cout << std::endl;

    if (!deviceId) {
        std::cout << "    deviceId is null → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    *deviceId = g_current_device;
    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtSetDevice(int32_t deviceId) {
    std::cout << "[aclrtSetDevice] deviceId=" << deviceId << std::endl;

    if (deviceId < 0 || deviceId >= g_device_count) {
        std::cout << "    invalid deviceId → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    g_current_device = deviceId;

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtMemcpy(void *dst,
                                         size_t destMax,
                                         const void *src,
                                         size_t count,
                                         aclrtMemcpyKind kind) {
    std::cout << "[aclrtMemcpy] ";
    log_ptr("dst", dst);
    std::cout << " ";
    log_ptr("src", src);
    std::cout << " count=" << count
              << " destMax=" << destMax
              << " kind=" << kind
              << std::endl;

    if (!dst || !src) {
        std::cout << "    null pointer → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    if (count > destMax) {
        std::cout << "    count > destMax → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    memcpy(dst, src, count);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtMemcpyAsync(void *dst,
                                              size_t destMax,
                                              const void *src,
                                              size_t count,
                                              aclrtMemcpyKind kind,
                                              aclrtStream stream) {
    std::cout << "[aclrtMemcpyAsync] ";
    log_ptr("dst", dst);
    std::cout << " ";
    log_ptr("src", src);
    std::cout << " count=" << count
              << " destMax=" << destMax
              << " kind=" << kind << " ";
    log_ptr("stream", stream);
    std::cout << std::endl;

    if (!dst || !src) {
        std::cout << "    null pointer → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    if (count > destMax) {
        std::cout << "    count > destMax → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    memcpy(dst, src, count);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtFree(void *devPtr) {
    std::cout << "[aclrtFree] ";
    log_ptr("devPtr", devPtr);
    std::cout << std::endl;

    free(devPtr);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtDestroyStream(aclrtStream stream) {
    std::cout << "[aclrtDestroyStream] ";
    log_ptr("stream", stream);
    std::cout << std::endl;

    // not‑NPU: стримы не хранятся, просто no-op

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtProcessReport(int32_t timeout) {
    std::cout << "[aclrtProcessReport] timeout=" << timeout << std::endl;

    // not‑NPU: no-op, Torch-NPU не использует отчёты

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtEventElapsedTime(float *ms, aclrtEvent startEvent, aclrtEvent endEvent) {
    std::cout << "[aclrtEventElapsedTime] ";
    log_ptr("ms", ms);
    log_ptr("startEvent", startEvent);
    log_ptr("endEvent", endEvent);
    std::cout << std::endl;

    if (!ms) {
        std::cout << "    ms is null → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    // not‑NPU: Torch-NPU не использует реальные события, возвращаем фиктивное время
    *ms = 0.1f;

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream) {
    std::cout << "[aclrtRecordEvent] ";
    log_ptr("event", event);
    log_ptr("stream", stream);
    std::cout << std::endl;

    // not‑NPU: Torch-NPU не использует реальные события, no-op

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t *value) {
    std::cout << "[aclrtGetDeviceInfo] deviceId=" << deviceId << " attr=" << attr << " ";
    log_ptr("value", value);
    std::cout << std::endl;

    if (!value) {
        std::cout << "    value is null → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    switch (attr) {
        case ACL_DEV_ATTR_AICPU_CORE_NUM:
            *value = 0; // нет AICPU
            break;

        case ACL_DEV_ATTR_AICORE_CORE_NUM:
            *value = info_ai_core_num;
            break;

        case ACL_DEV_ATTR_CUBE_CORE_NUM:
            *value = info_cube_core_num;
            break;

        case ACL_DEV_ATTR_VECTOR_CORE_NUM:
            *value = info_vector_core_num;
            break;

        case ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE:
            *value = 64 * 1024; // 64 KiB, stub
            break;

        case ACL_DEV_ATTR_TOTAL_GLOBAL_MEM_SIZE:
            *value = info_GM_size;
            break;

        case ACL_DEV_ATTR_L2_CACHE_SIZE:
            *value = info_L2_size;
            break;

        case ACL_DEV_ATTR_SMP_ID:
            *value = 0;
            break;

        case ACL_DEV_ATTR_PHY_CHIP_ID:
            *value = 0;
            break;

        case ACL_DEV_ATTR_SUPER_POD_DEVIDE_ID:
            *value = 0;
            break;

        case ACL_DEV_ATTR_SUPER_POD_SERVER_ID:
            *value = 0;
            break;

        case ACL_DEV_ATTR_SUPER_POD_ID:
            *value = 0;
            break;

        case ACL_DEV_ATTR_CUST_OP_PRIVILEGE:
            *value = 1;
            break;

        case ACL_DEV_ATTR_MAINBOARD_ID:
            *value = 0;
            break;

        case ACL_DEV_ATTR_IS_VIRTUAL:
            *value = 0;
            break;

        default:
            std::cout << "    unknown attr → ACL_ERROR_INVALID_PARAM" << std::endl;
            return ACL_ERROR_INVALID_PARAM;
    }

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclFinalize() {
    std::cout << "[aclFinalize]" << std::endl;

    // not‑NPU: no-op, Torch-NPU не требует финализации

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtDestroyEvent(aclrtEvent event) {
    std::cout << "[aclrtDestroyEvent] ";
    log_ptr("event", event);
    std::cout << std::endl;

    // not‑NPU: no-op
    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtDeviceCanAccessPeer(int32_t *canAccessPeer,
                                                      int32_t deviceId,
                                                      int32_t peerDeviceId) {
    std::cout << "[aclrtDeviceCanAccessPeer] ";
    log_ptr("canAccessPeer", canAccessPeer);
    std::cout << " deviceId=" << deviceId
              << " peerDeviceId=" << peerDeviceId << std::endl;

    if (!canAccessPeer) {
        std::cout << "    canAccessPeer is null → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    if (deviceId < 0 || deviceId >= (int)g_device_count ||
        peerDeviceId < 0 || peerDeviceId >= (int)g_device_count) {
        std::cout << "    invalid deviceId/peerDeviceId → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    *canAccessPeer = 0;

    std::cout << "    peer access unsupported → canAccessPeer=0" << std::endl;

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtMemset(void *devPtr,
                                         size_t maxCount,
                                         int32_t value,
                                         size_t count) {
    std::cout << "[aclrtMemset] ";
    log_ptr("devPtr", devPtr);
    std::cout << " maxCount=" << maxCount
              << " value=" << value
              << " count=" << count << std::endl;

    if (!devPtr) {
        std::cout << "    devPtr is null → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    if (count > maxCount) {
        std::cout << "    count > maxCount → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    // Torch‑NPU не использует асинхронность, не проверяет host/device,
    // не требует выравнивания — обычный memset полностью корректен.
    memset(devPtr, value, count);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclmdlInitDump() {
    std::cout << "[aclmdlInitDump]" << std::endl;

    // not‑NPU: no-op, Torch-NPU не использует dump pipeline

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclmdlFinalizeDump() {
    std::cout << "[aclmdlFinalizeDump]" << std::endl;

    // not‑NPU: no-op, Torch-NPU не использует dump pipeline

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclmdlSetDump(const char *dumpCfgPath) {
    std::cout << "[aclmdlSetDump] ";
    log_ptr("dumpCfgPath", dumpCfgPath);
    std::cout << std::endl;

    // not‑NPU: no-op, Torch-NPU не использует dump config

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event) {
    std::cout << "[aclrtStreamWaitEvent] ";
    log_ptr("stream", stream);
    log_ptr("event", event);
    std::cout << std::endl;

    // not‑NPU: Torch-NPU не использует реальные события и не ждёт их.
    // Полный no-op.

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtSynchronizeEvent(aclrtEvent event) {
    std::cout << "[aclrtSynchronizeEvent] ";
    log_ptr("event", event);
    std::cout << std::endl;

    // not‑NPU: Torch-NPU не использует реальные события.
    // Полный no-op.

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtCreateStream(aclrtStream *stream) {
    std::cout << "[aclrtCreateStream] ";
    log_ptr("stream_ptr", stream);
    std::cout << std::endl;

    if (!stream) {
        std::cout << "    stream_ptr is null → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    // not‑NPU: создаём фиктивный указатель, чтобы Torch‑NPU был доволен
    void *fake_stream = malloc(1);

    *stream = fake_stream;

    std::cout << "    created stream ";
    log_ptr("stream", fake_stream);
    std::cout << std::endl;

    return ACL_SUCCESS;
}

static aclrtContext g_current_context = nullptr;

ACL_FUNC_VISIBILITY aclError aclrtSetCurrentContext(aclrtContext context) {
    std::cout << "[aclrtSetCurrentContext] ";
    log_ptr("context", context);
    std::cout << std::endl;

    if (!context) {
        std::cout << "    context is null → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    g_current_context = context;

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtGetCurrentContext(aclrtContext *context) {
    std::cout << "[aclrtGetCurrentContext] ";
    log_ptr("context_ptr", context);
    std::cout << std::endl;

    if (!context) {
        std::cout << "    context_ptr is null → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    // Если контекст ещё не создан — создаём фиктивный
    if (!g_current_context) {
        g_current_context = malloc(1);
        std::cout << "    created default fake context ";
        log_ptr("ctx", g_current_context);
        std::cout << std::endl;
    }

    *context = g_current_context;

    std::cout << "    returned context ";
    log_ptr("ctx", g_current_context);
    std::cout << std::endl;

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t *free, size_t *total) {
    std::cout << "[aclrtGetMemInfo] attr=" << attr << " ";
    log_ptr("free_ptr", free);
    log_ptr("total_ptr", total);
    std::cout << std::endl;

    if (!free || !total) {
        std::cout << "    null pointer → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    // not‑NPU: Torch-NPU не различает DDR/HBM/huge/P2P.
    // Возвращаем общий объём памяти устройства.
    *total = info_GM_size;

    // not‑NPU: считаем, что вся память свободна.
    *free = info_GM_size;

    std::cout << "    free=" << *free << " total=" << *total << std::endl;

    return ACL_SUCCESS;
}

static bool g_acl_initialized = false;

ACL_FUNC_VISIBILITY aclError aclInit(const char *configPath) {
    std::cout << "[aclInit] ";
    log_ptr("configPath", configPath);
    std::cout << std::endl;

    if (g_acl_initialized) {
        std::cout << "    already initialized → ACL_SUCCESS" << std::endl;
        return ACL_SUCCESS;
    }

    g_acl_initialized = true;

    std::cout << "    initialized ACL runtime (not‑NPU stub)" << std::endl;

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtDeviceEnablePeerAccess(int32_t peerDeviceId, uint32_t flags) {
    std::cout << "[aclrtDeviceEnablePeerAccess] peerDeviceId=" << peerDeviceId
              << " flags=" << flags << std::endl;

    if (peerDeviceId < 0 || peerDeviceId >= (int)g_device_count) {
        std::cout << "    invalid peerDeviceId → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    if (flags != 0) {
        std::cout << "    flags must be zero (ignored in not‑NPU)" << std::endl;
        // не возвращаем ошибку — Torch‑NPU не умеет её обрабатывать
    }

    // not‑NPU: P2P не поддерживается, но enable должен возвращать успех
    std::cout << "    P2P unsupported → no-op" << std::endl;

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtResetEvent(aclrtEvent event, aclrtStream stream) {
    std::cout << "[aclrtResetEvent] ";
    log_ptr("event", event);
    std::cout << " ";
    log_ptr("stream", stream);
    std::cout << std::endl;

    // not‑NPU: Torch-NPU не использует реальные события.
    // Полный no-op.

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtResetDevice(int32_t deviceId) {
    std::cout << "[aclrtResetDevice] deviceId=" << deviceId << std::endl;

    if (deviceId < 0 || deviceId >= (int)g_device_count) {
        std::cout << "    invalid deviceId → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    // not‑NPU: реального устройства нет, контексты/стримы фиктивные.
    // Ничего не делаем, просто считаем, что reset прошёл успешно.

    std::cout << "    reset device (no-op in not‑NPU)" << std::endl;

    return ACL_SUCCESS;
}

typedef enum aclrtFloatOverflowMode {
    ACL_RT_OVERFLOW_MODE_SATURATION = 0,
    ACL_RT_OVERFLOW_MODE_INFNAN,
    ACL_RT_OVERFLOW_MODE_UNDEF,
} aclrtFloatOverflowMode;

static aclrtFloatOverflowMode g_sat_mode = ACL_RT_OVERFLOW_MODE_SATURATION;

ACL_FUNC_VISIBILITY aclError aclrtSetDeviceSatMode(aclrtFloatOverflowMode mode) {
    std::cout << "[aclrtSetDeviceSatMode] mode=" << mode << std::endl;

    // not‑NPU: просто сохраняем значение, чтобы Get мог вернуть то же самое
    g_sat_mode = mode;

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtGetDeviceSatMode(aclrtFloatOverflowMode *mode) {
    std::cout << "[aclrtGetDeviceSatMode] ";
    log_ptr("mode_ptr", mode);
    std::cout << std::endl;

    if (!mode) {
        std::cout << "    mode_ptr is null → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    *mode = g_sat_mode;

    std::cout << "    returned mode=" << *mode << std::endl;

    return ACL_SUCCESS;
}

typedef enum {
  ACL_OPT_DETERMINISTIC = 0,
  ACL_OPT_ENABLE_DEBUG_KERNEL = 1,
  ACL_OPT_STRONG_CONSISTENCY = 2,
  ACL_OPT_EARLY_START = 3
} aclSysParamOpt;

ACL_FUNC_VISIBILITY aclError aclrtCtxSetSysParamOpt(aclSysParamOpt opt, int64_t value) {
    std::cout << "[aclrtCtxSetSysParamOpt] opt=" << opt
              << " value=" << value << std::endl;

    // not‑NPU: системных параметров нет, просто принимаем и игнорируем
    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtSetOpExecuteTimeOut(uint32_t timeout) {
    std::cout << "[aclrtSetOpExecuteTimeOut] timeout=" << timeout << " sec" << std::endl;

    // not‑NPU: таймауты не поддерживаются, но Torch‑NPU ожидает успех
    return ACL_SUCCESS;
}



// ~~~ cann-ge-executor/ge-executor/include/acl/acl_base_mdl.h ~~~

ACL_FUNC_VISIBILITY aclTensorDesc *aclCreateTensorDesc(aclDataType dataType,
                                                       int numDims,
                                                       const int64_t *dims,
                                                       aclFormat format) {
    std::cout << "[aclCreateTensorDesc] ";
    std::cout << " dtype=" << static_cast<int>(dataType)
              << " numDims=" << numDims
              << " ";
    log_ptr("dims", dims);
    std::cout << " format=" << format << std::endl;

    if (numDims < 0 || (numDims > 0 && !dims)) {
        std::cout << "    invalid dims → nullptr" << std::endl;
        return nullptr;
    }

    aclTensorDesc *desc = new aclTensorDesc();
    if (!desc) {
        std::cout << "    new failed → nullptr" << std::endl;
        return nullptr;
    }

    desc->dtype = dataType;
    desc->format = format;

    desc->dims.clear();
    desc->dims.reserve(numDims);
    for (int i = 0; i < numDims; i++)
        desc->dims.push_back(dims[i]);

    std::cout << "    created tensor desc ";
    log_ptr("ptr", desc);
    std::cout << " dims=" << numDims << std::endl;

    return desc;
}

ACL_FUNC_VISIBILITY void aclDestroyTensorDesc(const aclTensorDesc *desc) {
    std::cout << "[aclDestroyTensorDesc] ";
    log_ptr("desc", desc);
    std::cout << std::endl;

    delete desc;
}

ACL_DEPRECATED_MESSAGE("aclGetTensorDescDim is deprecated, use aclGetTensorDescDimV2 instead")
ACL_FUNC_VISIBILITY int64_t aclGetTensorDescDim(const aclTensorDesc *desc, size_t index) {
    std::cout << "[aclGetTensorDescDim] ";
    log_ptr("desc", desc);
    std::cout << " index=" << index << std::endl;

    if (!desc) {
        std::cout << "    desc is null → -1" << std::endl;
        return -1;
    }

    if (index >= desc->dims.size()) {
        std::cout << "    index out of range → -1" << std::endl;
        return -1;
    }

    int64_t v = desc->dims[index];

    std::cout << "    dim=" << v << std::endl;

    return v;
}

ACL_FUNC_VISIBILITY aclError aclGetTensorDescDimV2(const aclTensorDesc *desc,
                                                   size_t index,
                                                   int64_t *dimSize) {
    std::cout << "[aclGetTensorDescDimV2] ";
    log_ptr("desc", desc);
    log_ptr("dimSize", dimSize);
    std::cout << " index=" << index << std::endl;

    if (!desc || !dimSize) {
        std::cout << "    invalid argument → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    if (index >= desc->dims.size()) {
        std::cout << "    index out of range → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    *dimSize = desc->dims[index];

    std::cout << "    dim=" << *dimSize << std::endl;

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclFormat aclGetTensorDescFormat(const aclTensorDesc *desc) {
    std::cout << "[aclGetTensorDescFormat] ";
    log_ptr("desc", desc);
    std::cout << std::endl;

    if (!desc) {
        std::cout << "    desc is null → ACL_FORMAT_UNDEFINED" << std::endl;
        return ACL_FORMAT_UNDEFINED;
    }

    aclFormat fmt = desc->format;

    std::cout << "    format=" << fmt << std::endl;

    return fmt;
}

ACL_FUNC_VISIBILITY size_t aclGetTensorDescNumDims(const aclTensorDesc *desc) {
    std::cout << "[aclGetTensorDescNumDims] ";
    log_ptr("desc", desc);
    std::cout << std::endl;

    if (!desc) {
        std::cout << "    desc is null → 0" << std::endl;
        return 0;
    }

    if (desc->dims.size() == 1 && desc->dims[0] == -2) {
        std::cout << "    dims = [-2] → ACL_UNKNOWN_RANK" << std::endl;
        return ACL_UNKNOWN_RANK;  // -2
    }

    size_t n = desc->dims.size();

    std::cout << "    numDims=" << n << std::endl;

    return n;
}

ACL_FUNC_VISIBILITY aclDataType aclGetTensorDescType(const aclTensorDesc *desc) {
    std::cout << "[aclGetTensorDescType] ";
    log_ptr("desc", desc);
    std::cout << std::endl;

    if (!desc) {
        std::cout << "    desc is null → ACL_DT_UNDEFINED" << std::endl;
        return ACL_DT_UNDEFINED;
    }

    aclDataType t = desc->dtype;

    std::cout << "    dtype=" << static_cast<int>(t) << std::endl;

    return t;
}

ACL_FUNC_VISIBILITY void aclSetTensorDescName(aclTensorDesc *desc, const char *name) {
    std::cout << "[aclSetTensorDescName] ";
    log_ptr("desc", desc);
    log_ptr("name", name);
    std::cout << std::endl;

    if (!desc || !name) {
        std::cout << "    invalid argument → ignored" << std::endl;
        return;
    }

    desc->name = name;

    std::cout << "    stored name=\"" << desc->name << "\"" << std::endl;
}

ACL_FUNC_VISIBILITY aclError aclSetTensorFormat(aclTensorDesc *desc, aclFormat format) {
    std::cout << "[aclSetTensorFormat] ";
    log_ptr("desc", desc);
    std::cout << " format=" << format << std::endl;

    if (!desc) {
        std::cout << "    desc is null → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    desc->format = format;

    std::cout << "    stored format=" << format << std::endl;

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclSetTensorPlaceMent(aclTensorDesc *desc, aclMemType memType) {
    std::cout << "[aclSetTensorPlaceMent] ";
    log_ptr("desc", desc);
    std::cout << " memType=" << memType << std::endl;

    if (!desc) {
        std::cout << "    desc is null → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    desc->memType = memType;

    std::cout << "    stored memType=" << memType << std::endl;

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclSetTensorShape(aclTensorDesc *desc, int numDims, const int64_t *dims) {
    std::cout << "[aclSetTensorShape] ";
    log_ptr("desc", desc);
    std::cout << " numDims=" << numDims << " ";
    log_ptr("dims", dims);
    std::cout << std::endl;

    if (!desc || !dims || numDims < 0) {
        std::cout << "    invalid argument → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    desc->dims.clear();
    desc->dims.reserve(numDims);

    for (int i = 0; i < numDims; i++)
        desc->dims.push_back(dims[i]);

    std::cout << "    stored shape: dims=" << numDims << std::endl;

    return ACL_SUCCESS;
}



// ~~~ cann-npu-runtime/runtime/pkg_inc/profiling/aprof_pub.h ~~~

#if (defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER))
#define MSVP_PROF_API __declspec(dllexport)
#else
#define MSVP_PROF_API __attribute__((visibility("default")))
#endif

#define MSPROF_REPORT_DATA_MAGIC_NUM 0x5A5AU
#define MSPROF_COMPACT_INFO_DATA_LENGTH 40
typedef void* VOID_PTR;

MSVP_PROF_API uint64_t MsprofGetHashId(const char *hashInfo, size_t length) {
    std::cout << "[MsprofGetHashId] ";
    log_ptr("hashInfo", hashInfo);
    std::cout << " length=" << length << std::endl;

    // простой детерминированный FNV‑1a hash
    uint64_t h = 1469598103934665603ULL;
    const uint8_t *p = reinterpret_cast<const uint8_t*>(hashInfo);
    for (size_t i = 0; i < length; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }

    std::cout << "    h = " << h << std::endl;

    return h;
}

struct MsprofApi { // for MsprofReportApi
#ifdef __cplusplus
    uint16_t magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
#else
    uint16_t magicNumber;
#endif
    uint16_t level;
    uint32_t type;
    uint32_t threadId;
    uint32_t reserve;
    uint64_t beginTime;
    uint64_t endTime;
    uint64_t itemId;
};

MSVP_PROF_API int32_t MsprofReportApi(uint32_t nonPersistantFlag, const struct MsprofApi *api) {
    std::cout << "[MsprofReportApi] nonPersistantFlag=" << nonPersistantFlag << " ";
    log_ptr("api", api);
    std::cout << std::endl;

    if (!api) {
        std::cout << "    api is null → FAILED" << std::endl;
        return -1;
    }

    // not‑NPU: заполняем структуру здесь же (как будто драйвер это делает)
    MsprofApi filled = *api;

    // thread id
    filled.threadId = static_cast<uint32_t>(
        std::hash<std::thread::id>{}(std::this_thread::get_id())
    );

    // timestamps
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
    uint64_t t = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
    filled.beginTime = t;
    filled.endTime = t + 100; // фиктивная длительность 100 ns

    // itemId = FNV‑1a(type + level)
    {
        uint64_t h = 1469598103934665603ULL;
        uint64_t x = (static_cast<uint64_t>(filled.type) << 32) | filled.level;
        for (int i = 0; i < 8; ++i) {
            h ^= (x >> (i * 8)) & 0xFF;
            h *= 1099511628211ULL;
        }
        filled.itemId = h;
    }

    // reserve всегда 0
    filled.reserve = 0;

    // magicNumber уже установлен конструктором
    std::cout << "    filled: magic=" << filled.magicNumber
              << " level=" << filled.level
              << " type=" << filled.type
              << " threadId=" << filled.threadId
              << " begin=" << filled.beginTime
              << " end=" << filled.endTime
              << " itemId=" << filled.itemId
              << std::endl;

    // not‑NPU: no-op
    return 0;
}

struct MsprofNodeBasicInfo {
    uint64_t opName;
    uint32_t taskType;
    uint64_t opType;
    uint32_t blockDim;
    uint32_t opFlag;
};

struct MsprofStreamExpandSpecInfo {
    uint8_t expandStatus;
    uint8_t reserve1;
    uint16_t reserve2;
};

struct MsprofHCCLOPInfo {  // for MsprofReportCompactInfo buffer data
    uint8_t relay : 1;     // Communication
    uint8_t retry : 1;     // Retransmission flag
    uint8_t dataType;      // Consistent with Type HcclDataType preservation
    uint64_t algType;      // The algorithm used by the communication operator, the hash key, whose value is a string separated by "-".
    uint64_t count;        // Number of data sent
    uint64_t groupName;    // group hash id
};

struct MsprofRuntimeTrack {  // for MsprofReportCompactInfo buffer data
    uint16_t deviceId;
    uint16_t streamId;
    uint32_t taskId;
    uint64_t taskType;       // task message hash id
    uint64_t kernelName;     // kernelname hash id
};

struct MsprofCaptureStreamInfo {  // for MsprofReportCompactInfo buffer data
    uint16_t captureStatus;     // Whether the mark is destroyed: 0 indicates normal, 1 indicates destroyed.
    uint16_t modelStreamId;     // capture stream id. Destroy the stream ID of the record, set it to UINT16_MAX.
    uint16_t originalStreamId;  // ori stream id. Destroy the stream ID of the record, set it to UINT16_MAX.
    uint16_t modelId;           // capture model id, independent of GE
    uint16_t deviceId;
};

struct MsprofDpuTrack {  // for MsprofReportCompactInfo buffer data
    uint16_t deviceId;   // high 4 bits, devType: dpu: 1, low 12 bits device id
    uint16_t streamId;
    uint32_t taskId;
    uint32_t taskType;    // task type enum
    uint32_t res;
    uint64_t startTime;   // start time
};

struct MsprofCompactInfo {  // for MsprofReportCompactInfo buffer data
#ifdef __cplusplus
    uint16_t magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
#else
    uint16_t magicNumber;
#endif
    uint16_t level;
    uint32_t type;
    uint32_t threadId;
    uint32_t dataLen;
    uint64_t timeStamp;
    union {
        uint8_t info[MSPROF_COMPACT_INFO_DATA_LENGTH];
        struct MsprofRuntimeTrack runtimeTrack;
        struct MsprofCaptureStreamInfo captureStreamInfo;
        struct MsprofNodeBasicInfo nodeBasicInfo;
        struct MsprofHCCLOPInfo hcclopInfo;
        struct MsprofDpuTrack dpuTack;
        struct MsprofStreamExpandSpecInfo streamExpandInfo;
    } data;
};

MSVP_PROF_API int32_t MsprofReportCompactInfo(uint32_t nonPersistantFlag, const VOID_PTR data, uint32_t length) {
    std::cout << "[MsprofReportCompactInfo] nonPersistantFlag=" << nonPersistantFlag
              << " length=" << length << " ";
    log_ptr("data", data);
    std::cout << std::endl;

    if (!data) {
        std::cout << "    data is null → FAILED" << std::endl;
        return -1;
    }

    if (length < sizeof(MsprofCompactInfo)) {
        std::cout << "    length < sizeof(MsprofCompactInfo) → FAILED" << std::endl;
        return -1;
    }

    const MsprofCompactInfo *src = reinterpret_cast<const MsprofCompactInfo*>(data);
    MsprofCompactInfo filled = *src;

    // thread id
    filled.threadId = static_cast<uint32_t>(
        std::hash<std::thread::id>{}(std::this_thread::get_id())
    );

    // timestamp
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
    filled.timeStamp = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();

    // magic number (если вдруг не установлен)
    filled.magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;

    // dataLen — оставляем как есть, Torch-NPU не проверяет
    // union data — не трогаем, просто логируем

    std::cout << "    filled: magic=" << filled.magicNumber
              << " level=" << filled.level
              << " type=" << filled.type
              << " threadId=" << filled.threadId
              << " dataLen=" << filled.dataLen
              << " timeStamp=" << filled.timeStamp
              << std::endl;

    // Логируем первые 32 байта info[] для отладки
    std::cout << "    data[0..31]: ";
    for (int i = 0; i < 32; ++i) {
        uint8_t b = filled.data.info[i];
        const char *hex = "0123456789abcdef";
        char hi = hex[(b >> 4) & 0xF];
        char lo = hex[b & 0xF];
        std::cout << hi << lo << " ";
    }
    std::cout << std::endl;

    // not‑NPU: no-op
    return 0;
}

MSVP_PROF_API uint64_t MsprofSysCycleTime(void) {
    std::cout << "[MsprofSysCycleTime]" << std::endl;

    // not‑NPU: фиктивное значение 1 ns

    return 1;
}



// ~~~ cann-npu-runtime/runtime/pkg_inc/runtime/runtime/kernel.h ~~~

#define RTS_DLL_EXPORT

#ifndef RTS_API
#ifdef RTS_DLL_EXPORT
#define RTS_API __attribute__((visibility("default")))
#else
#define RTS_API
#endif
#endif

typedef int32_t rtError_t;
static const int32_t RT_ERROR_NONE = 0; // success
static const int32_t RT_ERROR_INVALID_VALUE = 1; // ???

typedef struct tagRtDevBinary {
    uint32_t magic;    // magic number
    uint32_t version;  // version of binary
    const void *data;  // binary data
    uint64_t length;   // binary length
} rtDevBinary_t;

RTS_API rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **hdl) {
    std::cout << "[rtDevBinaryRegister] ";
    log_ptr("bin", bin);
    log_ptr("hdl", hdl);
    std::cout << std::endl;

    if (!bin || !hdl) {
        std::cout << "    bin or hdl is null → RT_ERROR_INVALID_VALUE" << std::endl;
        return RT_ERROR_INVALID_VALUE;
    }

    std::cout << "    rtDevBinary_t:" << std::endl;
    std::cout << "        magic=" << bin->magic << std::endl;
    std::cout << "        version=" << bin->version << std::endl;
    std::cout << "        data=";
    log_ptr("data", bin->data);
    std::cout << "        length=" << bin->length << std::endl;

    // not‑NPU: не загружаем бинарь, не парсим ELF, не регистрируем ничего.
    // Просто возвращаем фиктивный handle, чтобы последующие вызовы могли его логировать.
    *hdl = (void*)bin;

    std::cout << "    handle set to bin (fake handle)" << std::endl;

    return RT_ERROR_NONE;
}

#ifndef char_t
typedef char char_t;
#endif

RTS_API rtError_t rtFunctionRegister(void *binHandle, const void *stubFunc, const char_t *stubName,
                                     const void *kernelInfoExt, uint32_t funcMode) {
    std::cout << "[rtFunctionRegister] ";
    log_ptr("binHandle", binHandle);
    log_ptr("stubFunc", stubFunc);
    log_ptr("stubName", stubName);
    log_ptr("kernelInfoExt", kernelInfoExt);
    std::cout << " funcMode=" << funcMode;
    std::cout << std::endl;

    if (!binHandle || !stubFunc || !stubName) {
        std::cout << "    invalid argument → RT_ERROR_INVALID_VALUE" << std::endl;
        return RT_ERROR_INVALID_VALUE;
    }

    // not‑NPU: Torch-NPU не использует реальную регистрацию функций.
    // Мы просто логируем и возвращаем успех.

    return RT_ERROR_NONE;
}

typedef void *rtStream_t;

typedef struct tagRtSmData {
    uint64_t L2_mirror_addr;          // preload or swap source addr
    uint32_t L2_data_section_size;    // every data size
    uint8_t L2_preload;               // 1 - preload from mirrorAddr, 0 - no preload
    uint8_t modified;                 // 1 - data will be modified by kernel, 0 - no modified
    uint8_t priority;                 // data priority
    int8_t prev_L2_page_offset_base;  // remap source section offset
    uint8_t L2_page_offset_base;      // remap destination section offset
    uint8_t L2_load_to_ddr;           // 1 - need load out, 0 - no need
    uint8_t reserved[2];              // reserved
} rtSmData_t;

typedef struct tagRtSmCtrl {
    rtSmData_t data[8];  // data description
    uint64_t size;       // max page Num
    uint8_t remap[64];   /* just using for static remap mode, default:0xFF
                          array index: virtual l2 page id, array value: physic l2 page id */
    uint8_t l2_in_main;  // 0-DDR, 1-L2, default:0xFF
    uint8_t reserved[3];
} rtSmDesc_t;

RTS_API rtError_t rtKernelLaunch(const void *stubFunc, uint32_t numBlocks, void *args, uint32_t argsSize,
                                 rtSmDesc_t *smDesc, rtStream_t stm) {
    std::cout << "[rtKernelLaunch] ";
    log_ptr("stubFunc", stubFunc);
    std::cout << " numBlocks=" << numBlocks;
    log_ptr("args", args);
    std::cout << " argsSize=" << argsSize;
    log_ptr("smDesc", smDesc);
    log_ptr("stream", stm);
    std::cout << std::endl;

    if (!stubFunc) {
        std::cout << "    stubFunc is null → RT_ERROR_INVALID_VALUE" << std::endl;
        return RT_ERROR_INVALID_VALUE;
    }

    // Логируем smDesc, если он есть
    if (smDesc) {
        std::cout << "    rtSmDesc_t:" << std::endl;
        std::cout << "        size=" << smDesc->size << std::endl;
        std::cout << "        l2_in_main=" << (int)smDesc->l2_in_main << std::endl;

        std::cout << "        remap[0..7]: ";
        for (int i = 0; i < 8; i++) {
            std::cout << (int)smDesc->remap[i] << " ";
        }
        std::cout << std::endl;

        for (int i = 0; i < 8; i++) {
            const rtSmData_t &d = smDesc->data[i];
            std::cout << "        data[" << i << "] L2_mirror_addr=" << d.L2_mirror_addr
                      << " L2_data_section_size=" << d.L2_data_section_size
                      << " L2_preload=" << (int)d.L2_preload
                      << " modified=" << (int)d.modified
                      << " priority=" << (int)d.priority
                      << " prev_L2_page_offset_base=" << (int)d.prev_L2_page_offset_base
                      << " L2_page_offset_base=" << (int)d.L2_page_offset_base
                      << " L2_load_to_ddr=" << (int)d.L2_load_to_ddr
                      << std::endl;
        }
    }

    // not‑NPU: Torch-NPU не использует реальный запуск ядра.
    std::cout << "    kernel launched (fake)" << std::endl;

    return RT_ERROR_NONE;
}



// ~~~ cann-ge-executor/ge-executor/include/acl/acl_op.h ~~~

ACL_FUNC_VISIBILITY aclopAttr *aclopCreateAttr() {
    std::cout << "[aclopCreateAttr]" << std::endl;

    aclopAttr *a = new aclopAttr();
    if (!a) {
        std::cout << "    new failed → nullptr" << std::endl;
        return nullptr;
    }

    std::cout << "    created attr ";
    log_ptr("ptr", a);
    std::cout << std::endl;

    return a;
}

ACL_FUNC_VISIBILITY void aclopDestroyAttr(const aclopAttr *attr) {
    std::cout << "[aclopDestroyAttr] ";
    log_ptr("attr", attr);
    std::cout << std::endl;

    delete attr;
}

ACL_FUNC_VISIBILITY aclError aclopSetAttrBool(aclopAttr *attr, const char *attrName, uint8_t attrValue) {
    std::cout << "[aclopSetAttrBool] ";
    log_ptr("attr", attr);
    log_ptr("attrName", attrName);
    std::cout << " attrValue=" << (int)attrValue << std::endl;

    if (!attr || !attrName) {
        std::cout << "    invalid argument → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    attr->bools[attrName] = attrValue;

    std::cout << "    stored bool: " << attrName << "=" << (int)attrValue << std::endl;

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclopSetAttrDataType(aclopAttr *attr, const char *attrName, aclDataType attrValue) {
    std::cout << "[aclopSetAttrDataType] ";
    log_ptr("attr", attr);
    log_ptr("attrName", attrName);
    std::cout << " dtype=" << static_cast<int>(attrValue) << std::endl;

    if (!attr || !attrName) {
        std::cout << "    invalid argument → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    attr->dtypes[attrName] = attrValue;

    std::cout << "    stored dtype: " << attrName << "=" << static_cast<int>(attrValue) << std::endl;

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclopSetAttrFloat(aclopAttr *attr, const char *attrName, float attrValue) {
    std::cout << "[aclopSetAttrFloat] ";
    log_ptr("attr", attr);
    log_ptr("attrName", attrName);
    std::cout << " value=" << attrValue << std::endl;

    if (!attr || !attrName) {
        std::cout << "    invalid argument → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    attr->floats[attrName] = attrValue;

    std::cout << "    stored float: " << attrName << "=" << attrValue << std::endl;

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclopSetAttrInt(aclopAttr *attr, const char *attrName, int64_t attrValue) {
    std::cout << "[aclopSetAttrInt] ";
    log_ptr("attr", attr);
    log_ptr("attrName", attrName);
    std::cout << " value=" << attrValue << std::endl;

    if (!attr || !attrName) {
        std::cout << "    invalid argument → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    attr->ints[attrName] = attrValue;

    std::cout << "    stored int: " << attrName << "=" << attrValue << std::endl;

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclopSetAttrListBool(aclopAttr *attr, const char *attrName, int numValues,
                                                  const uint8_t *values) {
    std::cout << "[aclopSetAttrListBool] ";
    log_ptr("attr", attr);
    log_ptr("attrName", attrName);
    log_ptr("values", values);
    std::cout << " numValues=" << numValues << std::endl;

    if (!attr || !attrName || !values || numValues < 0) {
        std::cout << "    invalid argument → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    std::vector<uint8_t> out;
    out.reserve(numValues);

    for (int i = 0; i < numValues; i++)
        out.push_back(values[i]);

    attr->list_bools[attrName] = std::move(out);

    std::cout << "    stored list<uint8_t>: " << attrName
              << " size=" << numValues << std::endl;

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclopSetAttrListFloat(aclopAttr *attr, const char *attrName, int numValues,
                                                   const float *values) {
    std::cout << "[aclopSetAttrListFloat] ";
    log_ptr("attr", attr);
    log_ptr("attrName", attrName);
    log_ptr("values", values);
    std::cout << " numValues=" << numValues << std::endl;

    if (!attr || !attrName || !values || numValues < 0) {
        std::cout << "    invalid argument → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    std::vector<float> out;
    out.reserve(numValues);

    for (int i = 0; i < numValues; i++)
        out.push_back(values[i]);

    attr->list_floats[attrName] = std::move(out);

    std::cout << "    stored list<float>: " << attrName
              << " size=" << numValues << std::endl;

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclopSetAttrListInt(aclopAttr *attr, const char *attrName, int numValues,
                                                 const int64_t *values) {
    std::cout << "[aclopSetAttrListInt] ";
    log_ptr("attr", attr);
    log_ptr("attrName", attrName);
    log_ptr("values", values);
    std::cout << " numValues=" << numValues << std::endl;

    if (!attr || !attrName || !values || numValues < 0) {
        std::cout << "    invalid argument → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    std::vector<int64_t> out;
    out.reserve(numValues);

    for (int i = 0; i < numValues; i++)
        out.push_back(values[i]);

    attr->list_ints[attrName] = std::move(out);

    std::cout << "    stored list<int64_t>: " << attrName
              << " size=" << numValues << std::endl;

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclopSetAttrListListInt(aclopAttr *attr,
                                                     const char *attrName,
                                                     int numLists,
                                                     const int *numValues,
                                                     const int64_t *const values[]) {
    std::cout << "[aclopSetAttrListListInt] ";
    log_ptr("attr", attr);
    log_ptr("attrName", attrName);
    log_ptr("numValues", numValues);
    log_ptr("values", values);
    std::cout << " numLists=" << numLists << std::endl;

    if (!attr || !attrName || !numValues || !values || numLists < 0) {
        std::cout << "    invalid argument → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    std::vector<std::vector<int64_t>> out;
    out.reserve(numLists);

    for (int i = 0; i < numLists; i++) {
        int n = numValues[i];
        const int64_t *src = values[i];

        if (!src || n < 0) {
            std::cout << "    invalid inner list → ACL_ERROR_INVALID_PARAM" << std::endl;
            return ACL_ERROR_INVALID_PARAM;
        }

        std::vector<int64_t> inner;
        inner.reserve(n);

        for (int j = 0; j < n; j++) {
            inner.push_back(src[j]);
        }

        out.push_back(std::move(inner));
    }

    attr->list_list_ints[attrName] = std::move(out);

    std::cout << "    stored list<list<int64_t>>: " << attrName
              << " lists=" << numLists << std::endl;

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclopSetAttrString(aclopAttr *attr, const char *attrName, const char *attrValue) {
    std::cout << "[aclopSetAttrString] ";
    log_ptr("attr", attr);
    log_ptr("attrName", attrName);
    log_ptr("attrValue", attrValue);
    std::cout << std::endl;

    if (!attr || !attrName || !attrValue) {
        std::cout << "    invalid argument → ACL_ERROR_INVALID_PARAM" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    attr->strings[attrName] = attrValue;

    std::cout << "    stored string: " << attrName
              << "=\"" << attrValue << "\"" << std::endl;

    return ACL_SUCCESS;
}



#ifdef __cplusplus
}
#endif
EOF

g++ -shared -fPIC lib/not_acl.cpp -o lib/libascendcl.so

echo "Created lib/libascendcl.so"



#if [ ! -f lib/libacl_op_compiler.so ]; then
    cat > lib/not_acl_op_compiler.cpp << EOF
#include <iostream>
$common

void __not_acl_op_compiler_placeholder() {}


#ifdef __cplusplus
extern "C" {
#endif


// cann-ge-compiler/ge-compiler/include/acl/acl_op_compiler.h

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
    std::cout << "[aclopCompileAndExecute] ";
    log_ptr("opType", opType);
    std::cout << " numInputs=" << numInputs
              << " numOutputs=" << numOutputs << " ";
    log_ptr("stream", stream);
    std::cout << std::endl;

    // not‑NPU: Torch-NPU не использует ACL Op Compiler,
    // не читает результаты, не проверяет ошибки.
    // Полный no-op.

    return ACL_SUCCESS;
}

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

ACL_FUNC_VISIBILITY aclError aclSetCompileopt(aclCompileOpt opt, const char *value) {
    std::cout << "[aclSetCompileopt] opt=" << opt << " ";
    log_ptr("value", value);
    if (value) {
        std::cout << " value_str=\"" << value << "\"";
    }
    std::cout << std::endl;

    // not‑NPU: компилятора нет, все compile options игнорируются
    return ACL_SUCCESS;
}


#ifdef __cplusplus
}
#endif
EOF

    g++ -shared -fPIC lib/not_acl_op_compiler.cpp -o lib/libacl_op_compiler.so

    echo "Created lib/libacl_op_compiler.so"
#fi



if [ ! -f lib/libge_runner.so ]; then
    cat > lib/not_ge_runner.c << 'EOF'
void __not_ge_runner_placeholder() {}
EOF

    gcc -shared -fPIC lib/not_ge_runner.c -o lib/libge_runner.so

    echo "Created lib/libge_runner.so"
fi



if [ ! -f lib/libgraph.so ]; then
    cat > lib/not_graph.c << 'EOF'
void __not_graph_placeholder() {}
EOF

    gcc -shared -fPIC lib/not_graph.c -o lib/libgraph.so

    echo "Created lib/libgraph.so"
fi



if [ ! -f lib/libacl_tdt_channel.so ]; then
    cat > lib/not_acl_tdt_channel.c << 'EOF'
void __not_acl_tdt_channel_placeholder() {}
EOF

    gcc -shared -fPIC lib/not_acl_tdt_channel.c -o lib/libacl_tdt_channel.so

    echo "Created lib/libacl_tdt_channel.so"
fi



# export LD_LIBRARY_PATH="$HOME/not_npu/lib:$LD_LIBRARY_PATH"
# ~/not_npu/build_not_npu.sh && python -c "import torch_npu"   # stage 1, чтобы просто завелись библиотеки .so внутри torch_npu
# ~/not_npu/build_not_npu.sh && python -c "import torch; print(torch.randn(2, 3, device='npu'))"



: << 'COMMENT'
# общий поиск либы и заголовков по символу:

LIB_BASE="$HOME/tmp/Ascend-cann/run_package"
SYMBOL=aclAppLog

for f in $(find $LIB_BASE -name '*.so*'); do
    nm -D "$f" 2>/dev/null | grep -F $SYMBOL && echo " → $f"
done
grep -Rnw $LIB_BASE -e $SYMBOL --include='*.h' 2>/dev/null


# точечный поиск по либам и заголовкам:

LIBS=(
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/lib/libruntime.so
    cann-npu-runtime/runtime/lib/libprofapi.so
    cann-ge-compiler/ge-compiler/lib64/libacl_op_compiler.so
)
for f in ${LIBS[*]}; do
    nm -D "$LIB_BASE/$f" 2>/dev/null | grep -F $SYMBOL && echo " → $f"
done

INCLUDES=(
    cann-npu-runtime/runtime/include/external/acl/acl_base_rt.h
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
    cann-npu-runtime/runtime/pkg_inc/profiling/aprof_pub.h
    cann-npu-runtime/runtime/pkg_inc/runtime/runtime/kernel.h
    cann-ge-executor/ge-executor/include/acl/acl_base_mdl.h
    cann-ge-executor/ge-executor/include/acl/acl_op.h
    cann-ge-compiler/ge-compiler/include/acl/acl_op_compiler.h
)
for f in ${INCLUDES[*]}; do
    grep -F "$SYMBOL(" "$LIB_BASE/$f" 2>/dev/null && echo " → $f"
done


# сборщик символов

SYM_BASE=$(for f in $(find $LIB_BASE -name '*.so*'); do
    nm -D --defined-only "$f" 2>/dev/null \
        | awk '$3 !~ /_/ && $3 !~ /@/ && $3 ~ /[A-Z]/ {print $3}'
done | sort -u)
echo "$SYM_BASE"

TORCH_NPU_LIBS=/opt/python311/lib/python3.11/site-packages/torch_npu/lib
REQ=$(for f in $(find $TORCH_NPU_LIBS -name '*.so*'); do
    nm -D --undefined-only "$f" 2>/dev/null \
        | awk '$NF !~ /_/ && $NF !~ /@/ && $NF ~ /[A-Z]/ {print $NF}'
done | sort -u)
echo "$REQ"

for sym in $REQ; do
    echo "$sym"
    for lib in ${LIBS[*]}; do
        if nm -D --defined-only "$LIB_BASE/$lib" 2>/dev/null | grep -qw "$sym"; then
            echo "    $lib"
        fi
    done
    for f in ${INCLUDES[*]}; do
        if grep -q "$sym(" "$LIB_BASE/$f" 2>/dev/null; then
            echo "    $f"
        fi
    done
    IFS=':' read -ra DIRS <<< "$LD_LIBRARY_PATH"
    unset available;
    for d in "${DIRS[@]}"; do
        for so in "$d"/*.so*; do
            [ -e "$so" ] || continue
            if nm -D --defined-only "$so" 2>/dev/null | grep -qw "$sym"; then
              # echo "    [ld.so] available via $so"
                available=''
            fi
        done
    done
    if [ ! -v available ]; then
        echo "    UNRELEASE"
    fi
done


# финальный результат анализа символов

MsprofGetHashId
    cann-npu-runtime/runtime/lib/libprofapi.so
    cann-npu-runtime/runtime/pkg_inc/profiling/aprof_pub.h
MsprofReportApi
    cann-npu-runtime/runtime/lib/libprofapi.so
    cann-npu-runtime/runtime/pkg_inc/profiling/aprof_pub.h
MsprofReportCompactInfo
    cann-npu-runtime/runtime/lib/libprofapi.so
    cann-npu-runtime/runtime/pkg_inc/profiling/aprof_pub.h
MsprofSysCycleTime
    cann-npu-runtime/runtime/lib/libprofapi.so
    cann-npu-runtime/runtime/pkg_inc/profiling/aprof_pub.h
ParameterClass
THPDeviceType   (здесь все *Type и *Class относятся к pytorch)
THPDtypeType
THPEventClass
THPStorageClass
THPStreamClass
THPVariableClass
aclAppLog
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_base_rt.h
aclCreateDataBuffer
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_base_rt.h
aclCreateTensorDesc
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_base_mdl.h
aclDestroyDataBuffer
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_base_rt.h
aclDestroyTensorDesc
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_base_mdl.h
aclFinalize
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclGetDeviceCapability
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_base_rt.h
aclGetRecentErrMsg
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclGetTensorDescDimV2
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_base_mdl.h
aclGetTensorDescFormat
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_base_mdl.h
aclGetTensorDescNumDims
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_base_mdl.h
aclGetTensorDescType
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_base_mdl.h
aclInit
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclSetTensorDescName
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_base_mdl.h
aclSetTensorFormat
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_base_mdl.h
aclSetTensorPlaceMent
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_base_mdl.h
aclSetTensorShape
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_base_mdl.h
aclmdlFinalizeDump
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclmdlInitDump
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclmdlSetDump
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclopCompileAndExecute
    cann-ge-compiler/ge-compiler/lib64/libacl_op_compiler.so
    cann-ge-compiler/ge-compiler/include/acl/acl_op_compiler.h
aclopCreateAttr
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_op.h
aclopDestroyAttr
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_op.h
aclopSetAttrBool
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_op.h
aclopSetAttrDataType
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_op.h
aclopSetAttrFloat
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_op.h
aclopSetAttrInt
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_op.h
aclopSetAttrListBool
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_op.h
aclopSetAttrListFloat
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_op.h
aclopSetAttrListInt
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_op.h
aclopSetAttrListListInt
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_op.h
aclopSetAttrString
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_op.h
aclrtCreateStream
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtDestroyEvent
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtDestroyStream
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtDeviceCanAccessPeer
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtDeviceEnablePeerAccess
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtEventElapsedTime
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtFree
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtFreeHost
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtGetCurrentContext
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtGetDevice
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtGetDeviceCount
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtGetDeviceInfo
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtGetMemInfo
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtMalloc
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtMallocHost
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtMemcpy
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtMemcpyAsync
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtMemset
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtProcessReport
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtRecordEvent
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtResetDevice
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtResetEvent
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtSetCurrentContext
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtSetDevice
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtStreamWaitEvent
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtSynchronizeEvent
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtSynchronizeStream
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
rtDevBinaryRegister
    cann-npu-runtime/runtime/lib/libruntime.so
    cann-npu-runtime/runtime/pkg_inc/runtime/runtime/kernel.h
rtFunctionRegister
    cann-npu-runtime/runtime/lib/libruntime.so
    cann-npu-runtime/runtime/pkg_inc/runtime/runtime/kernel.h
rtKernelLaunch
    cann-npu-runtime/runtime/lib/libruntime.so
    cann-npu-runtime/runtime/pkg_inc/runtime/runtime/kernel.h
COMMENT
