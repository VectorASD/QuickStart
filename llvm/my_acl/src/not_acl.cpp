#include "common.h"
#include "op_profiler.h"  // log_op_timings, reset_op_timings

#include <iostream>  // cout, endl
#include <cstdarg>   // va_end, va_list, va_start
#include <cstring>   // memcpy, memset, size_t
#include <thread>    // get_id, thread
#include <sstream>   // ostringstream

#ifndef NOT_ACL
#define NOT_ACL

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
    std::ostringstream log;
    log << "[aclAppLog] level=" << logLevel
        << " func=" << (func ? func : "<null>")
        << " file=" << (file ? file : "<null>")
        << " line=" << line << '\n';

    char buffer[8192];

    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    log << "\n    msg: " << buffer << '\n';
    log_output(log);
}

ACL_FUNC_VISIBILITY aclError aclGetDeviceCapability(uint32_t deviceId, aclDeviceInfo deviceInfo, int64_t *value) {
    std::ostringstream log;
    log << "[aclGetDeviceCapability] deviceId=" << deviceId
        << " deviceInfo=" << deviceInfo
        << " value_ptr=" << static_cast<const void*>(value);

    if (!value) {
        log << "\n    value is null → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
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
            log << "\n    unknown deviceInfo → ACL_ERROR_INVALID_PARAM";
            log_output(log, true);
            return ACL_ERROR_INVALID_PARAM;
    }

    log_output(log);
    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclDataBuffer *aclCreateDataBuffer(void *data, size_t size) {
    // Всегда работает в паре с aclCreateTensorDesc, т.к. есть сырой буфер, но нет данных о нём
    std::ostringstream log;
    log << "[aclCreateDataBuffer] data=" << data
        << " size=" << size;

    aclDataBuffer *buf = new aclDataBuffer();
    if (!buf) {
        log << "\n    new failed → nullptr";
        log_output(log);
        return nullptr;
    }

    buf->data = data;
    buf->size = size;

 // log << "\n    created aclDataBuffer ptr=" << buf;
 // log_output(log);
    return buf;
}

ACL_FUNC_VISIBILITY aclError aclDestroyDataBuffer(const aclDataBuffer *dataBuffer) {
 // std::ostringstream log;
 // log << "[aclDestroyDataBuffer] dataBuffer=" << dataBuffer;
 // log_output(log);

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
    std::ostringstream log;
    log << "[aclrtSynchronizeStream] stream=" << stream;
    log_output(log);

    // not‑NPU: стрим не хранит реальных задач, синхронизация не требуется
    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t timeout) {
    std::ostringstream log;
    log << "[aclrtSynchronizeStreamWithTimeout] stream=" << stream
        << " timeout=" << timeout
        << " ms";
 // log_output(log);

    // в рамках only-RAM машины нечего синхронизировать, вся память - это обычный malloc
    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtGetDeviceCount(uint32_t *count) {
    *count = g_device_count;
 // std::ostringstream log;
 // log << "[aclrtGetDeviceCount] count=" << *count;
 // log_output(log);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY const char *aclGetRecentErrMsg() {
    log_output("[aclGetRecentErrMsg]");
    return "not-npu: no recent error";
}

ACL_FUNC_VISIBILITY aclError aclrtFreeHost(void *hostPtr) {
    std::ostringstream log;
    log << "[aclrtFreeHost] hostPtr=" << hostPtr;
    log_output(log);

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
    std::ostringstream log;
    log << "[aclrtMalloc] size=" << size
        << " policy=" << policy
        << " devPtr=" << devPtr;

    if (!devPtr) {
        log << "\n    devPtr is null → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    // not‑NPU: device memory = обычный malloc
    void *ptr = malloc(size);
    *devPtr = ptr;

    log << "\n    allocated" << ptr;

    if (!ptr) {
        log << "\n    allocation failed → ACL_ERROR_BAD_ALLOC";
        log_output(log, true);
        return ACL_ERROR_BAD_ALLOC;
    }

    log_output(log);
    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtMallocHost(void **hostPtr, size_t size) {
    std::ostringstream log;
    log << "[aclrtMallocHost] size=" << size;

    void *ptr = malloc(size);
    *hostPtr = ptr;

    log << "\n    allocated=" << ptr;

    if (!ptr) {
        log << "\n    allocation failed → ACL_ERROR_BAD_ALLOC";
        log_output(log, true);
        return ACL_ERROR_BAD_ALLOC;
    }

    log_output(log);
    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtMallocAlign32(void **devPtr,
                                                size_t size,
                                                aclrtMemMallocPolicy policy) {
    std::ostringstream log;
    log << "[aclrtMallocAlign32] size=" << size
        << " policy=" << policy
        << " devPtr=" << devPtr;

    if (!devPtr) {
        log << "\n    devPtr is null → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    if (size == 0) {
        *devPtr = nullptr;
        log << "\n    size == 0 → devPtr = nullptr";
        log_output(log, true);
        return ACL_SUCCESS;
    }

    // not‑NPU: выделяем память, выровненную на 32 байта
    void *ptr = nullptr;
    int ret = posix_memalign(&ptr, 32, size);
    if (ret != 0 || !ptr) {
        log << "\n    aligned_alloc failed → ACL_ERROR_BAD_ALLOC";
        log_output(log, true);
        return ACL_ERROR_BAD_ALLOC;
    }

    *devPtr = ptr;

    log << "\n    allocated=" << ptr;
    log_output(log);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtGetDevice(int32_t *deviceId) {
    std::ostringstream log;
    log << "[aclrtGetDevice]";

    if (!deviceId) {
        log << "\n    deviceId is null → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }
 // log_output(log);

    *deviceId = g_current_device;

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtSetDevice(int32_t deviceId) {
    std::ostringstream log;
    log << "[aclrtSetDevice] deviceId=" << deviceId;

    if (deviceId < 0 || deviceId >= g_device_count) {
        log << "\n    invalid deviceId → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }
    log_output(log);

    g_current_device = deviceId;

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtMemcpy(void *dst,
                                         size_t destMax,
                                         const void *src,
                                         size_t count,
                                         aclrtMemcpyKind kind) {
    std::ostringstream log;
    log << "[aclrtMemcpy] dst=" << dst
        << " src=" << src
        << " count=" << count
        << " destMax=" << destMax
        << " kind=" << kind;

    if (!dst || !src) {
        log << "\n    null pointer → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    if (count > destMax) {
        log << "\n    count > destMax → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }
 // log_output(log);

    memcpy(dst, src, count);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtMemcpyAsync(void *dst,
                                              size_t destMax,
                                              const void *src,
                                              size_t count,
                                              aclrtMemcpyKind kind,
                                              aclrtStream stream) {
    std::ostringstream log;
    log << "[aclrtMemcpyAsync] dst=" << dst
        << " src=" << src
        << " count=" << count
        << " destMax=" << destMax
        << " kind=" << kind
        << " stream=" << stream;

    if (!dst || !src) {
        log << "\n    null pointer → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    if (count > destMax) {
        log << "\n    count > destMax → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }
    log_output(log);

    memcpy(dst, src, count);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtFree(void *devPtr) {
    std::ostringstream log;
    log << "[aclrtFree] devPtr=" << devPtr;
    log_output(log);

    free(devPtr);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtDestroyStream(aclrtStream stream) {
    std::ostringstream log;
    log << "[aclrtDestroyStream] stream=" << stream;
    log_output(log);

    free(stream);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtProcessReport(int32_t timeout) {
    std::ostringstream log;
    log << "[aclrtProcessReport] timeout=" << timeout;
    log_output(log);

    // not‑NPU: no-op, Torch-NPU не использует отчёты

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtEventElapsedTime(float *ms, aclrtEvent startEvent, aclrtEvent endEvent) {
    std::ostringstream log;
    log << "[aclrtEventElapsedTime] ms=" << ms
        << " startEvent=" << startEvent
        << " endEvent=" << endEvent;

    if (!ms) {
        log << "\n    ms is null → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }
    log_output(log);

    // not‑NPU: Torch-NPU не использует реальные события, возвращаем фиктивное время
    *ms = 0.1f;

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream) {
    std::ostringstream log;
    log << "[aclrtRecordEvent] event=" << event
        << " stream=" << stream;
    log_output(log);

    // not‑NPU: Torch-NPU не использует реальные события, no-op

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t *value) {
    std::ostringstream log;
    log << "[aclrtGetDeviceInfo] deviceId=" << deviceId
        << " attr=" << attr
        << " value=" << value;

    if (!value) {
        log << "\n    value is null → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
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
            log << "\n    unknown attr → ACL_ERROR_INVALID_PARAM";
            log_output(log, true);
            return ACL_ERROR_INVALID_PARAM;
    }

    log_output(log);
    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclFinalize() {
    log_output("[aclFinalize]");

    // not‑NPU: no-op, Torch-NPU не требует финализации

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtDestroyEvent(aclrtEvent event) {
    std::ostringstream log;
    log << "[aclrtDestroyEvent] event=" << event;
    log_output(log);

    // not‑NPU: no-op
    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtDeviceCanAccessPeer(int32_t *canAccessPeer,
                                                      int32_t deviceId,
                                                      int32_t peerDeviceId) {
    std::ostringstream log;
    log << "[aclrtDeviceCanAccessPeer] canAccessPeer=" << canAccessPeer
        << " deviceId=" << deviceId
        << " peerDeviceId=" << peerDeviceId;

    if (!canAccessPeer) {
        log << "\n    canAccessPeer is null → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    if (deviceId < 0 || deviceId >= (int)g_device_count ||
        peerDeviceId < 0 || peerDeviceId >= (int)g_device_count) {
        log << "\n    invalid deviceId/peerDeviceId → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    *canAccessPeer = 0;

    log << "\n    peer access unsupported → canAccessPeer=0";
    log_output(log);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtMemset(void *devPtr,
                                         size_t maxCount,
                                         int32_t value,
                                         size_t count) {
    std::ostringstream log;
    log << "[aclrtMemset] devPtr=" << devPtr
        << " maxCount=" << maxCount
        << " value=" << value
        << " count=" << count;

    if (!devPtr) {
        log << "\n    devPtr is null → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    if (count > maxCount) {
        log << "\n    count > maxCount → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }
    log_output(log);

    // Torch‑NPU не использует асинхронность, не проверяет host/device,
    // не требует выравнивания — обычный memset полностью корректен.
    memset(devPtr, value, count);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclmdlInitDump() {
    log_output("[aclmdlInitDump]");

    // not‑NPU: no-op, Torch-NPU не использует dump pipeline

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclmdlFinalizeDump() {
    log_output("[aclmdlFinalizeDump]");

    // not‑NPU: no-op, Torch-NPU не использует dump pipeline

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclmdlSetDump(const char *dumpCfgPath) {
    std::ostringstream log;
    log << "[aclmdlSetDump] dumpCfgPath=" << dumpCfgPath;
    log_output(log);

    // not‑NPU: no-op, Torch-NPU не использует dump config

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event) {
    std::ostringstream log;
    log << "[aclrtStreamWaitEvent] stream=" << stream
        << " event=" << event;
    log_output(log);

    // not‑NPU: Torch-NPU не использует реальные события и не ждёт их.
    // Полный no-op.

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtSynchronizeEvent(aclrtEvent event) {
    std::ostringstream log;
    log << "[aclrtSynchronizeEvent] event=" << event;
    log_output(log);

    // not‑NPU: Torch-NPU не использует реальные события.
    // Полный no-op.

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtCreateStream(aclrtStream *stream) {
    std::ostringstream log;
    log << "[aclrtCreateStream]";

    if (!stream) {
        log << "\n    stream_ptr is null → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    // not‑NPU: создаём фиктивный указатель, чтобы Torch‑NPU был доволен
    void *fake_stream = malloc(1);

    *stream = fake_stream;

    log << "\n    created stream=" << fake_stream;
    log_output(log);

    return ACL_SUCCESS;
}

static aclrtContext g_current_context = nullptr;

ACL_FUNC_VISIBILITY aclError aclrtSetCurrentContext(aclrtContext context) {
    std::ostringstream log;
    log << "[aclrtSetCurrentContext] context=" << context;

    if (!context) {
        log << "\n    context is null → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }
    log_output(log);

    g_current_context = context;

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtGetCurrentContext(aclrtContext *context) {
    std::ostringstream log;
    log << "[aclrtGetCurrentContext] context_ptr=" << context;

    if (!context) {
        log << "\n    context_ptr is null → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    // Если контекст ещё не создан — создаём фиктивный
    if (!g_current_context) {
        g_current_context = malloc(1);
        log << "\n    created default fake context";
    }

    *context = g_current_context;

    log << "\n    returned context=" << g_current_context;
    log_output(log);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t *free, size_t *total) {
    std::ostringstream log;
    log << "[aclrtGetMemInfo] attr=" << attr
        << " free_ptr=" << free
        << " total_ptr=" << total;

    if (!free || !total) {
        log << "\n    null pointer → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    // not‑NPU: Torch-NPU не различает DDR/HBM/huge/P2P.
    // Возвращаем общий объём памяти устройства.
    *total = info_GM_size;
    // not‑NPU: считаем, что вся память свободна.
    *free = info_GM_size;

    log << "\n    free=" << *free << " total=" << *total;
    log_output(log);

    return ACL_SUCCESS;
}

static bool g_acl_initialized = false;

ACL_FUNC_VISIBILITY aclError aclInit(const char *configPath) {
    std::ostringstream log;
    log << "[aclInit] configPath=" << configPath;

    if (g_acl_initialized) {
        log << "\n    already initialized → ACL_SUCCESS";
        log_output(log);
        return ACL_SUCCESS;
    }

    g_acl_initialized = true;

    log << "\n    initialized ACL runtime (not‑NPU stub)";
    log_output(log);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtDeviceEnablePeerAccess(int32_t peerDeviceId, uint32_t flags) {
    std::ostringstream log;
    log << "[aclrtDeviceEnablePeerAccess] peerDeviceId=" << peerDeviceId
        << " flags=" << flags;

    if (peerDeviceId < 0 || peerDeviceId >= (int)g_device_count) {
        log << "\n    invalid peerDeviceId → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    if (flags != 0) {
        log << "\n    flags must be zero (ignored in not‑NPU)";
        // не возвращаем ошибку — Torch‑NPU не умеет её обрабатывать
    }

    // not‑NPU: P2P не поддерживается, но enable должен возвращать успех
    log << "\n    P2P unsupported → no-op";
    log_output(log);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtResetEvent(aclrtEvent event, aclrtStream stream) {
    std::ostringstream log;
    log << "[aclrtResetEvent] event=" << event
        << " stream=" << stream;
    log_output(log);

    // not‑NPU: Torch-NPU не использует реальные события.
    // Полный no-op.

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtResetDevice(int32_t deviceId) {
    std::ostringstream log;
    log << "[aclrtResetDevice] deviceId=" << deviceId;

    if (deviceId < 0 || deviceId >= (int)g_device_count) {
        log << "\n    invalid deviceId → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    log << "\n    reset device (no-op in not‑NPU)";
    log_output(log);

    log_op_timings();
    reset_op_timings();

    return ACL_SUCCESS;
}

typedef enum aclrtFloatOverflowMode {
    ACL_RT_OVERFLOW_MODE_SATURATION = 0,
    ACL_RT_OVERFLOW_MODE_INFNAN,
    ACL_RT_OVERFLOW_MODE_UNDEF,
} aclrtFloatOverflowMode;

static aclrtFloatOverflowMode g_sat_mode = ACL_RT_OVERFLOW_MODE_SATURATION;

ACL_FUNC_VISIBILITY aclError aclrtSetDeviceSatMode(aclrtFloatOverflowMode mode) {
    std::ostringstream log;
    log << "[aclrtSetDeviceSatMode] mode=" << mode;
    log_output(log);

    // not‑NPU: просто сохраняем значение, чтобы Get мог вернуть то же самое
    g_sat_mode = mode;

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtGetDeviceSatMode(aclrtFloatOverflowMode *mode) {
    std::ostringstream log;
    log << "[aclrtGetDeviceSatMode] mode_ptr=" << mode;

    if (!mode) {
        log << "\n    mode_ptr is null → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    *mode = g_sat_mode;

    log << "\n    returned mode=" << *mode;
    log_output(log);

    return ACL_SUCCESS;
}

typedef enum {
  ACL_OPT_DETERMINISTIC = 0,
  ACL_OPT_ENABLE_DEBUG_KERNEL = 1,
  ACL_OPT_STRONG_CONSISTENCY = 2,
  ACL_OPT_EARLY_START = 3
} aclSysParamOpt;

ACL_FUNC_VISIBILITY aclError aclrtCtxSetSysParamOpt(aclSysParamOpt opt, int64_t value) {
    std::ostringstream log;
    log << "[aclrtCtxSetSysParamOpt] opt=" << opt
        << " value=" << value;
    log_output(log);

    // not‑NPU: системных параметров нет, просто принимаем и игнорируем
    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtSetOpExecuteTimeOut(uint32_t timeout) {
    std::ostringstream log;
    log << "[aclrtSetOpExecuteTimeOut] timeout=" << timeout << " sec";
    log_output(log);

    // not‑NPU: таймауты не поддерживаются, но Torch‑NPU ожидает успех
    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtSetStreamOverflowSwitch(aclrtStream stream, uint32_t flag) {
    std::ostringstream log;
    log << "[aclrtSetStreamOverflowSwitch] stream=" << stream
        << " flag=" << flag;
    log_output(log);

    // not‑NPU: no real stream overflow switch, just log and accept any flag
    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtDestroyStreamForce(aclrtStream stream) {
    std::ostringstream log;
    log << "[aclrtDestroyStreamForce] stream=" << stream;
    log_output(log);
    
    free(stream);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtSynchronizeDevice(void) {
    log_output("[aclrtSynchronizeDevice]");

    // not‑NPU: нет реального устройства, синхронизация не требуется
    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtSynchronizeDeviceWithTimeout(int32_t timeout) {
    std::ostringstream log;
    log << "[aclrtSynchronizeDeviceWithTimeout] timeout=" << timeout;
    log_output(log);

    // not‑NPU: нет реального устройства, синхронизация не требуется
    return ACL_SUCCESS;
}



// ~~~ cann-ge-executor/ge-executor/include/acl/acl_base_mdl.h ~~~

ACL_FUNC_VISIBILITY aclTensorDesc *aclCreateTensorDesc(aclDataType dataType,
                                                       int numDims,
                                                       const int64_t *dims,
                                                       aclFormat format) {
    // Всегда работает в паре с aclCreateDataBuffer, т.к. это даёт описание данных, но нет самого сырого буфера
    std::ostringstream log;
    log << "[aclCreateTensorDesc] dtype=" << static_cast<int>(dataType)
        << " numDims=" << numDims
        << " dims=" << dims
        << " format=" << format;

    if (numDims < 0 || (numDims > 0 && !dims)) {
        log << "\n    invalid dims → nullptr";
        log_output(log);
        return nullptr;
    }

    aclTensorDesc *desc = new aclTensorDesc();
    if (!desc) {
        log << "\n    new failed → nullptr";
        log_output(log);
        return nullptr;
    }

    desc->dtype = dataType;
    desc->format = format;

    desc->dims.clear();
    desc->dims.reserve(numDims);
    for (int i = 0; i < numDims; i++)
        desc->dims.push_back(dims[i]);

 // log << "\n    created tensor desc=" << desc
 //     << " dims=" << numDims;
 // log_output(log);

    return desc;
}

ACL_FUNC_VISIBILITY void aclDestroyTensorDesc(const aclTensorDesc *desc) {
 // std::ostringstream log;
 // log << "[aclDestroyTensorDesc] desc=" << desc;
 // log_output(log);

    delete desc;
}

ACL_DEPRECATED_MESSAGE("aclGetTensorDescDim is deprecated, use aclGetTensorDescDimV2 instead")
ACL_FUNC_VISIBILITY int64_t aclGetTensorDescDim(const aclTensorDesc *desc, size_t index) {
    std::ostringstream log;
    log << "[aclGetTensorDescDim] desc=" << desc
        << " index=" << index;

    if (!desc) {
        log << "\n    desc is null → -1";
        log_output(log);
        return -1;
    }

    if (index >= desc->dims.size()) {
        log << "\n    index out of range → -1";
        log_output(log);
        return -1;
    }

    int64_t v = desc->dims[index];

    log << "\n    dim=" << v;
    log_output(log);

    return v;
}

ACL_FUNC_VISIBILITY aclError aclGetTensorDescDimV2(const aclTensorDesc *desc,
                                                   size_t index,
                                                   int64_t *dimSize,
                                                   bool log_it = true) {
    std::ostringstream log;
    log << "[aclGetTensorDescDimV2] desc=" << desc
        << " dimSize" << dimSize
        << " index=" << index;

    if (!desc || !dimSize) {
        log << "\n    invalid argument → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    if (index >= desc->dims.size()) {
        log << "\n    index out of range → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    *dimSize = desc->dims[index];

    if (log_it) {
        log << "\n    dim=" << *dimSize;
        log_output(log);
    }

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclFormat aclGetTensorDescFormat(const aclTensorDesc *desc) {
    std::ostringstream log;
    log << "[aclGetTensorDescFormat] desc=" << desc;

    if (!desc) {
        log << "\n    desc is null → ACL_FORMAT_UNDEFINED";
        log_output(log);
        return ACL_FORMAT_UNDEFINED;
    }

    aclFormat fmt = desc->format;

    log << "\n    format=" << fmt;
    log_output(log);

    return fmt;
}

ACL_FUNC_VISIBILITY size_t aclGetTensorDescNumDims(const aclTensorDesc *desc,
                                                   bool log_it = true) {
    std::ostringstream log;
    log << "[aclGetTensorDescNumDims] desc=" << desc;

    if (!desc) {
        log << "\n    desc is null → 0";
        log_output(log);
        return 0;
    }

    if (desc->dims.size() == 1 && desc->dims[0] == -2) {
        log << "\n    dims = [-2] → ACL_UNKNOWN_RANK";
        log_output(log);
        return ACL_UNKNOWN_RANK;  // -2
    }

    size_t n = desc->dims.size();

    if (log_it) {
        log << "\n    numDims=" << n;
        log_output(log);
    }

    return n;
}

ACL_FUNC_VISIBILITY aclDataType aclGetTensorDescType(const aclTensorDesc *desc,
                                                     bool log_it = true) {
    std::ostringstream log;
    log << "[aclGetTensorDescType] desc=" << desc;

    if (!desc) {
        log << "\n    desc is null → ACL_DT_UNDEFINED";
        log_output(log);
        return ACL_DT_UNDEFINED;
    }

    aclDataType t = desc->dtype;

    if (log_it) {
        log << "\n    dtype=" << static_cast<int>(t);
        log_output(log);
    }

    return t;
}

ACL_FUNC_VISIBILITY void aclSetTensorDescName(aclTensorDesc *desc, const char *name) {
    std::ostringstream log;
    log << "[aclSetTensorDescName] desc=" << desc
        << " name=" << name;

    if (!desc || !name) {
        log << "\n    invalid argument → ignored";
        log_output(log);
        return;
    }

    desc->name = name;

 // log << "\n    stored name=\"" << desc->name << "\"";
 // log_output(log);
}

ACL_FUNC_VISIBILITY aclError aclSetTensorFormat(aclTensorDesc *desc, aclFormat format) {
    std::ostringstream log;
    log << "[aclSetTensorFormat] desc" << desc
        << " format=" << format;

    if (!desc) {
        log << "\n    desc is null → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    desc->format = format;

 // log << "\n    stored format=" << format;
 // log_output(log);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclSetTensorPlaceMent(aclTensorDesc *desc, aclMemType memType) {
    std::ostringstream log;
    log << "[aclSetTensorPlaceMent] desc=" << desc
        << " memType=" << aclMemTypeToString(memType);

    if (!desc) {
        log << "\n    desc is null → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    aclMemType oldMemType = desc->memType;
    // лениво перемещает данные тензора (aclDataBuffer) в нужную память
    // под lazy подразумевается, что это произойдёт только в будущем и только для выходов при вызове
    // aclopCompileAndExecute или подобных операций манипуляции с данными
    desc->memType = memType;

 // log << "\n    " << aclMemTypeToString(oldMemType) << " -> " << aclMemTypeToString(memType);
 // log_output(log);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclSetTensorShape(aclTensorDesc *desc, int numDims, const int64_t *dims) {
    std::ostringstream log;
    log << "[aclSetTensorShape] desc=" << desc
        << " numDims=" << numDims << " "
        << " dims=" << dims;

    if (!desc || !dims || numDims < 0) {
        log << "\n    invalid argument → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    desc->dims.clear();
    desc->dims.reserve(numDims);

    for (int i = 0; i < numDims; i++)
        desc->dims.push_back(dims[i]);

 // log << "\n    stored shape: dims=" << numDims;
 // log_output(log);

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
    std::ostringstream log;
    log << "[MsprofGetHashId] hashInfo=" << hashInfo
        << " length=" << length;

    // простой детерминированный FNV‑1a hash
    uint64_t h = 1469598103934665603ULL;
    const uint8_t *p = reinterpret_cast<const uint8_t*>(hashInfo);
    for (size_t i = 0; i < length; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }

    log << "\n    h = " << h;
    log_output(log);

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
    std::ostringstream log;
    log << "[MsprofReportApi] nonPersistantFlag=" << nonPersistantFlag
        << " api=" << api;

    if (!api) {
        log << "\n    api is null → FAILED";
        log_output(log);
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
    log << "\n    filled: magic=" << filled.magicNumber
        << " level=" << filled.level
        << " type=" << filled.type
        << " threadId=" << filled.threadId
        << " begin=" << filled.beginTime
        << " end=" << filled.endTime
        << " itemId=" << filled.itemId;
    log_output(log);

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
    std::ostringstream log;
    log << "[MsprofReportCompactInfo] nonPersistantFlag=" << nonPersistantFlag
        << " length=" << length
        << " data=" << data;

    if (!data) {
        log << "\n    data is null → FAILED";
        log_output(log);
        return -1;
    }

    if (length < sizeof(MsprofCompactInfo)) {
        log << "\n    length < sizeof(MsprofCompactInfo) → FAILED";
        log_output(log);
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

    log << "\n    filled: magic=" << filled.magicNumber
        << " level=" << filled.level
        << " type=" << filled.type
        << " threadId=" << filled.threadId
        << " dataLen=" << filled.dataLen
        << " timeStamp=" << filled.timeStamp;

    // Логируем первые 32 байта info[] для отладки
    log << "\n    data[0..31]: ";
    for (int i = 0; i < 32; ++i) {
        uint8_t b = filled.data.info[i];
        const char *hex = "0123456789abcdef";
        char hi = hex[(b >> 4) & 0xF];
        char lo = hex[b & 0xF];
        log << hi << lo << " ";
    }
    log_output(log);

    // not‑NPU: no-op
    return 0;
}

MSVP_PROF_API uint64_t MsprofSysCycleTime(void) {
    log_output("[MsprofSysCycleTime]");

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
    std::ostringstream log;
    log << "[rtDevBinaryRegister] bin=" << bin
        << " hdl=" << hdl;

    if (!bin || !hdl) {
        log << "\n    bin or hdl is null → RT_ERROR_INVALID_VALUE";
        log_output(log);
        return RT_ERROR_INVALID_VALUE;
    }

    log << "\n    rtDevBinary_t:"
        << "\n        magic=" << bin->magic
        << "\n        version=" << bin->version
        << "\n        data=" << bin->data
        << "\n        length=" << bin->length;

    // not‑NPU: не загружаем бинарь, не парсим ELF, не регистрируем ничего.
    // Просто возвращаем фиктивный handle, чтобы последующие вызовы могли его логировать.
    *hdl = (void*)bin;

    log << "\n    handle set to bin (fake handle)";
    log_output(log);

    return RT_ERROR_NONE;
}

#ifndef char_t
typedef char char_t;
#endif

RTS_API rtError_t rtFunctionRegister(void *binHandle, const void *stubFunc, const char_t *stubName,
                                     const void *kernelInfoExt, uint32_t funcMode) {
    std::ostringstream log;
    log << "[rtFunctionRegister] binHandle=" << binHandle
        << " stubFunc=" << stubFunc
        << " stubName=" << stubName
        << " kernelInfoExt=" << kernelInfoExt
        << " funcMode=" << funcMode;

    if (!binHandle || !stubFunc || !stubName) {
        log << "\n    invalid argument → RT_ERROR_INVALID_VALUE";
        log_output(log);
        return RT_ERROR_INVALID_VALUE;
    }
    log_output(log);

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
    std::ostringstream log;
    log << "[rtKernelLaunch] stubFunc=" << stubFunc
        << " numBlocks=" << numBlocks
        << " args=" << args
        << " argsSize=" << argsSize
        << " smDesc=" << smDesc
        << " stream=" << stm;

    if (!stubFunc) {
        log << "\n    stubFunc is null → RT_ERROR_INVALID_VALUE";
        log_output(log);
        return RT_ERROR_INVALID_VALUE;
    }

    // Логируем smDesc, если он есть
    if (smDesc) {
        log << "\n    rtSmDesc_t:"
            << "\n        size=" << smDesc->size
            << "\n        l2_in_main=" << (int)smDesc->l2_in_main

            << "\n        remap[0..7]: ";
        for (int i = 0; i < 8; i++) {
            log << (int)smDesc->remap[i] << " ";
        }

        for (int i = 0; i < 8; i++) {
            const rtSmData_t &d = smDesc->data[i];
            log << "\n        data[" << i << "]: L2_mirror_addr=" << d.L2_mirror_addr
                << " L2_data_section_size=" << d.L2_data_section_size
                << " L2_preload=" << (int)d.L2_preload
                << " modified=" << (int)d.modified
                << " priority=" << (int)d.priority
                << " prev_L2_page_offset_base=" << (int)d.prev_L2_page_offset_base
                << " L2_page_offset_base=" << (int)d.L2_page_offset_base
                << " L2_load_to_ddr=" << (int)d.L2_load_to_ddr;
        }
    }

    // not‑NPU: Torch-NPU не использует реальный запуск ядра.
    log << "\n    kernel launched (fake)";
    log_output(log);

    return RT_ERROR_NONE;
}



// ~~~ cann-ge-executor/ge-executor/include/acl/acl_op.h ~~~

ACL_FUNC_VISIBILITY aclopAttr *aclopCreateAttr() {
    std::ostringstream log;
    log << "[aclopCreateAttr]";

    aclopAttr *a = new aclopAttr();
    if (!a) {
        log << "\n    new failed → nullptr";
        return nullptr;
    }

 // log << "\n    created attr=" << a;
 // log_output(log);

    return a;
}

ACL_FUNC_VISIBILITY void aclopDestroyAttr(const aclopAttr *attr) {
 // std::ostringstream log;
 // log << "[aclopDestroyAttr] attr=" << attr;
 // log_output(log);

    delete attr;
}

ACL_FUNC_VISIBILITY aclError aclopSetAttrBool(aclopAttr *attr, const char *attrName, uint8_t attrValue) {
    std::ostringstream log;
    log << "[aclopSetAttrBool] attr=" << attr
        << " attrName=" << attrName
        << " attrValue=" << (int)attrValue;

    if (!attr || !attrName) {
        log << "\n    invalid argument → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    attr->bools[attrName] = attrValue;

 // log << "\n    stored bool: " << attrName << "=" << (int)attrValue;
 // log_output(log);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclopSetAttrDataType(aclopAttr *attr, const char *attrName, aclDataType attrValue) {
    std::ostringstream log;
    log << "[aclopSetAttrDataType] attr=" << attr
        << " attrName=" << attrName
        << " dtype=" << static_cast<int>(attrValue);

    if (!attr || !attrName) {
        log << "\n    invalid argument → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    attr->dtypes[attrName] = attrValue;

 // log << "\n    stored dtype: " << attrName << "=" << static_cast<int>(attrValue);
 // log_output(log);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclopSetAttrFloat(aclopAttr *attr, const char *attrName, float attrValue) {
    std::ostringstream log;
    log << "[aclopSetAttrFloat] attr=" << attr
        << " attrName=" << attrName
        << " value=" << attrValue;

    if (!attr || !attrName) {
        log << "\n    invalid argument → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    attr->floats[attrName] = attrValue;

 // log << "\n    stored float: " << attrName << "=" << attrValue;
 // log_output(log);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclopSetAttrInt(aclopAttr *attr, const char *attrName, int64_t attrValue) {
    std::ostringstream log;
    log << "[aclopSetAttrInt] attr=" << attr
        << " attrName=" << attrName
        << " value=" << attrValue;

    if (!attr || !attrName) {
        log << "\n    invalid argument → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    attr->ints[attrName] = attrValue;

 // log << "\n    stored int: " << attrName << "=" << attrValue;
 // log_output(log);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclopSetAttrListBool(aclopAttr *attr, const char *attrName, int numValues,
                                                  const uint8_t *values) {
    std::ostringstream log;
    log << "[aclopSetAttrListBool] attr=" << attr
        << " attrName=" << attrName
        << " values=" << values
        << " numValues=" << numValues;

    if (!attr || !attrName || !values || numValues < 0) {
        log << "\n    invalid argument → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    std::vector<uint8_t> out;
    out.reserve(numValues);

    for (int i = 0; i < numValues; i++)
        out.push_back(values[i]);

    attr->list_bools[attrName] = std::move(out);

 // log << "\n    stored list<uint8_t>: " << attrName
 //     << " size=" << numValues;
 // log_output(log);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclopSetAttrListFloat(aclopAttr *attr, const char *attrName, int numValues,
                                                   const float *values) {
    std::ostringstream log;
    log << "[aclopSetAttrListFloat] attr=" << attr
        << " attrName=" << attrName
        << " values=" << values
        << " numValues=" << numValues;

    if (!attr || !attrName || !values || numValues < 0) {
        log << "\n    invalid argument → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    std::vector<float> out;
    out.reserve(numValues);

    for (int i = 0; i < numValues; i++)
        out.push_back(values[i]);

    attr->list_floats[attrName] = std::move(out);

 // log << "\n    stored list<float>: " << attrName
 //     << " size=" << numValues;
 // log_output(log);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclopSetAttrListInt(aclopAttr *attr, const char *attrName, int numValues,
                                                 const int64_t *values) {
    std::ostringstream log;
    log << "[aclopSetAttrListInt] attr=" << attr
        << " attrName=" << attrName
        << " values=" << values
        << " numValues=" << numValues;

    if (!attr || !attrName || !values || numValues < 0) {
        log << "\n    invalid argument → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    std::vector<int64_t> out;
    out.reserve(numValues);

    for (int i = 0; i < numValues; i++)
        out.push_back(values[i]);

    attr->list_ints[attrName] = std::move(out);

 // log << "\n    stored list<int64_t>: " << attrName
 //     << " size=" << numValues;
 // log_output(log);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclopSetAttrListListInt(aclopAttr *attr,
                                                     const char *attrName,
                                                     int numLists,
                                                     const int *numValues,
                                                     const int64_t *const values[]) {
    std::ostringstream log;
    log << "[aclopSetAttrListListInt] attr=" << attr
        << " attrName=" << attrName
        << " numValues=" << numValues
        << " values=" << values
        << " numLists=" << numLists;

    if (!attr || !attrName || !numValues || !values || numLists < 0) {
        log << "\n    invalid argument → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    std::vector<std::vector<int64_t>> out;
    out.reserve(numLists);

    for (int i = 0; i < numLists; i++) {
        int n = numValues[i];
        const int64_t *src = values[i];

        if (!src || n < 0) {
            log << "\n    invalid inner list → ACL_ERROR_INVALID_PARAM";
            log_output(log, true);
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

 // log << "\n    stored list<list<int64_t>>: " << attrName
 //     << " lists=" << numLists;
 // log_output(log);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclopSetAttrString(aclopAttr *attr, const char *attrName, const char *attrValue) {
    std::ostringstream log;
    log << "[aclopSetAttrString] attr=" << attr
        << " attrName=" << attrName
        << " attrValue=" << attrValue;

    if (!attr || !attrName || !attrValue) {
        log << "\n    invalid argument → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    attr->strings[attrName] = attrValue;

 // log << "\n    stored string: " << attrName
 //     << "=\"" << attrValue << "\"";
 // log_output(log);

    return ACL_SUCCESS;
}



#ifdef __cplusplus
}
#endif

#endif // NOT_ACL
