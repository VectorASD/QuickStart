#include "common.h"
#include "op_profiler.h"  // log_op_timings, reset_op_timings
#include "../env/include/runtime/runtime/rt.h"
#include "../env/include/acl/acl.h"

#include <iostream>  // cout, endl
#include <cstdarg>   // va_end, va_list, va_start
#include <cstring>   // memcpy, memset, size_t
#include <thread>    // get_id, thread
#include <sstream>   // ostringstream

#include <sys/mman.h>  // MAP_FAILED, MAP_SHARED, PROT_READ, PROT_WRITE, mmap and 3 more
#include <sys/stat.h>  // fstat, stat
#include <fcntl.h>     // O_CREAT, O_RDONLY, O_RDWR
#include <unistd.h>    // close, ftruncate


#ifndef NOT_ACL
#define NOT_ACL

void __not_acl_placeholder() {}


#ifdef __cplusplus
extern "C" {
#endif

static int64_t info_L2_size         = 112LL * 1024 * 1024;               // 112 MiB
static int64_t info_GM_size         = (79LL * 1024 + 736) * 1024 * 1024; // 79 Gb, 736 Mb


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

    uint32_t aiCoreCnt;

    switch (deviceInfo) {
        case ACL_DEVICE_INFO_AI_CORE_NUM:
            if (rtGetAiCoreCount(&aiCoreCnt))
                return ACL_ERROR_FAILURE;
            *value = aiCoreCnt;
            break;
        case ACL_DEVICE_INFO_VECTOR_CORE_NUM:
            if (rtGetAiCoreCount(&aiCoreCnt))
                return ACL_ERROR_FAILURE;
            *value = aiCoreCnt * 2;
            break;
        case ACL_DEVICE_INFO_L2_SIZE:
            *value = info_L2_size;
            break;
        default:
            log << "\n    unknown deviceInfo → ACL_ERROR_INVALID_PARAM";
            log_output(log, true);
            return ACL_ERROR_INVALID_PARAM;
    }

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
    buf->owned = false;

 // log << "\n    created aclDataBuffer ptr=" << buf;
 // log_output(log);
    return buf;
}

ACL_FUNC_VISIBILITY aclError aclDestroyDataBuffer(const aclDataBuffer *dataBuffer) {
 // std::ostringstream log;
 // log << "[aclDestroyDataBuffer] dataBuffer=" << dataBuffer;
 // log_output(log);

    if (dataBuffer->owned)
        free(dataBuffer->data);
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
    rtError_t ret = rtGetDeviceCount(reinterpret_cast<int32_t*>(count));
    return ret ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}
ACL_FUNC_VISIBILITY aclError aclrtGetDevice(int32_t *deviceId) {
    rtError_t ret = rtGetDevice(deviceId);
    return ret ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}
ACL_FUNC_VISIBILITY aclError aclrtSetDevice(int32_t deviceId) {
    rtError_t ret = rtSetDevice(deviceId);
    return ret ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}
ACL_FUNC_VISIBILITY const char *aclrtGetSocName() {
    static char buffer[64];
    rtGetSocVersion(buffer, sizeof(buffer));
    return buffer;
}

ACL_FUNC_VISIBILITY aclError aclrtMallocHost(void **hostPtr, size_t size) {
    switch (rtMallocHost(hostPtr, size, 0)) {
        case RT_ERROR_NONE:          return ACL_SUCCESS;
        case RT_ERROR_INVALID_VALUE: return ACL_ERROR_FAILURE;
        case RT_ERROR_BAD_ALLOC:     return ACL_ERROR_BAD_ALLOC;
        default:
            return ACL_ERROR_INTERNAL_ERROR;
    }
}
ACL_FUNC_VISIBILITY aclError aclrtFreeHost(void *hostPtr) {
    rtError_t ret = rtFreeHost(hostPtr);
    return ret ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtMalloc(void **devPtr,
                                         size_t size,
                                         aclrtMemMallocPolicy policy) {
    if (!devPtr)
        return ACL_ERROR_INVALID_PARAM;
    switch(rtMalloc(devPtr, size, 0, 0)) {
        case RT_ERROR_NONE:          return ACL_SUCCESS;
        case RT_ERROR_INVALID_VALUE: return ACL_ERROR_FAILURE;
        case RT_ERROR_BAD_ALLOC:     return ACL_ERROR_BAD_ALLOC;
        default:
            return ACL_ERROR_INTERNAL_ERROR;
    }
}
ACL_FUNC_VISIBILITY aclError aclrtFree(void *devPtr) {
    rtError_t ret = rtFree(devPtr);
    return ret ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtMemcpy(void *dst,
                                         size_t destMax,
                                         const void *src,
                                         size_t count,
                                         aclrtMemcpyKind kind) {
    rtError_t ret = rtMemcpy(dst, destMax, src, count, static_cast<rtMemcpyKind_t>(kind));
    return ret ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}


ACL_FUNC_VISIBILITY const char *aclGetRecentErrMsg() {
    log_output("[aclGetRecentErrMsg]");
    return "not-npu: no recent error";
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

ACL_FUNC_VISIBILITY aclError aclrtProcessReport(int32_t timeout) {
    std::ostringstream log;
    log << "[aclrtProcessReport] timeout=" << timeout;
    log_output(log);

    // not‑NPU: no-op, Torch-NPU не использует отчёты

    return ACL_SUCCESS;
}


ACL_FUNC_VISIBILITY aclError aclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag) {
    std::ostringstream log;
    log << "[aclrtCreateEventWithFlag] flag=" << flag;

    if (!event) {
        log << "\n    event_ptr is null → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    // not‑NPU: создаём фиктивный event, чтобы Torch‑NPU был доволен
    void *not_event = malloc(1);
    if (!not_event) {
        log << "\n    malloc failed → ACL_ERROR_BAD_ALLOC";
        log_output(log, true);
        return ACL_ERROR_BAD_ALLOC;
    }

    *event = not_event;
    log << "\n    created event=" << not_event;
    log_output(log);

    return ACL_SUCCESS;
}
ACL_FUNC_VISIBILITY aclError aclrtCreateEvent(aclrtEvent *event) {
    return aclrtCreateEventWithFlag(event, 0);
}
ACL_FUNC_VISIBILITY aclError aclrtCreateEventExWithFlag(aclrtEvent *event, uint32_t flag) {
    return aclrtCreateEventWithFlag(event, flag);
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

ACL_FUNC_VISIBILITY aclError aclrtDestroyEvent(aclrtEvent event) {
    std::ostringstream log;
    log << "[aclrtDestroyEvent] event=" << event;
    log_output(log);

    // not‑NPU: no-op
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

ACL_FUNC_VISIBILITY aclError aclrtResetEvent(aclrtEvent event, aclrtStream stream) {
    std::ostringstream log;
    log << "[aclrtResetEvent] event=" << event
        << " stream=" << stream;
    log_output(log);

    // not‑NPU: Torch-NPU не использует реальные события.
    // Полный no-op.

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

    uint32_t aiCoreCnt;

    switch (attr) {
        case ACL_DEV_ATTR_AICPU_CORE_NUM:
            *value = 0; // нет AICPU
            break;

        case ACL_DEV_ATTR_AICORE_CORE_NUM:
            if (rtGetAiCoreCount(&aiCoreCnt))
                return ACL_ERROR_FAILURE;
            *value = aiCoreCnt;
            break;

        case ACL_DEV_ATTR_CUBE_CORE_NUM:
            if (rtGetAiCoreCount(&aiCoreCnt))
                return ACL_ERROR_FAILURE;
            *value = aiCoreCnt;
            break;

        case ACL_DEV_ATTR_VECTOR_CORE_NUM:
            if (rtGetAiCoreCount(&aiCoreCnt))
                return ACL_ERROR_FAILURE;
            *value = aiCoreCnt * 2;
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

    int32_t device_count;
    rtGetDeviceCount(&device_count);

    if (deviceId < 0 || deviceId >= device_count ||
        peerDeviceId < 0 || peerDeviceId >= device_count) {
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

ACL_FUNC_VISIBILITY aclError aclrtCreateStream(aclrtStream *stream) {
    rtError_t ret = rtStreamCreate(stream, 0);
    return ret ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}
ACL_FUNC_VISIBILITY aclError aclrtDestroyStream(aclrtStream stream) {
    rtError_t ret = rtStreamDestroy(stream);
    return ret ? ACL_ERROR_FAILURE : ACL_SUCCESS;
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
        if (!g_current_context) {
            log << "\n    malloc failed → ACL_ERROR_BAD_ALLOC";
            log_output(log, true);
            return ACL_ERROR_BAD_ALLOC;
        }
        log << "\n    created default NOT context";
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

    /*auto home = get_home_path();
    auto main_dir = home / "not_npu";
    auto setenv = main_dir / "setenv.sh";

    std::string ccec_content = R"(#!/bin/bash
if [ "$1" = "-print-targets" ]
    then echo "hiipu64"
    else echo "not bisheng OK"
fi
)";
    ensure_file(main_dir / "ccec", ccec_content, true);

    std::string install_content = "version=9.1.0"; // Нужно для get_cann_version из /opt/python311/lib/python3.11/site-packages/torch_npu/utils/collect_env.py
    ensure_file(main_dir / "ascend_toolkit_install.info", install_content);

    auto main_dir_str = "\"" + main_dir.string() + "\"";
    std::string setenv_content =
        "export TRITON_NPU_COMPILER_PATH=" + main_dir_str + "\n"
        "export ASCEND_HOME_PATH=" + main_dir_str + "\n"
        "export ASCEND_OPP_PATH=" + main_dir_str + "\n";
    ensure_file(setenv, setenv_content);

    ensure_directory(main_dir / "runtime");
    ensure_directory(main_dir / "compiler");

    auto setenv_str = "\"" + setenv.string() + "\"";
    std::string bashrc_addition =
        "# NOT_NPU: source custom rc file\n"
        "if [ -f " + setenv_str + " ]; then\n"
        "    source " + setenv_str + "\n"
        "fi";

    const char* env_val = std::getenv("TRITON_NPU_COMPILER_PATH");
    if (ensure_file_contains(home / ".bashrc", bashrc_addition) || !env_val || !env_val[0]) {
        std::cerr << ANSI_RED << "[NOT_NPU] ~/.bashrc has been updated. "
                  << "Please restart your terminal or run 'source ~/.bashrc' "
                  << "for the environment to take effect in new sessions."
                  << ANSI_RESET << std::endl;
        // return ACL_ERROR_UNINITIALIZE; как много слов...
        // exit(1); Вызывает pybind11::handle::dec_ref за GIL, что разносит систему
        _exit(1);
    }*/

    g_acl_initialized = true;

    log << "\n    initialized ACL runtime";
    log_output(log);

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtDeviceEnablePeerAccess(int32_t peerDeviceId, uint32_t flags) {
    std::ostringstream log;
    log << "[aclrtDeviceEnablePeerAccess] peerDeviceId=" << peerDeviceId
        << " flags=" << flags;

    int32_t device_count;
    rtGetDeviceCount(&device_count);

    if (peerDeviceId < 0 || peerDeviceId >= device_count) {
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

ACL_FUNC_VISIBILITY aclError aclrtResetDevice(int32_t deviceId) {
    std::ostringstream log;
    log << "[aclrtResetDevice] deviceId=" << deviceId;

    int32_t device_count;
    rtGetDeviceCount(&device_count);

    if (deviceId < 0 || deviceId >= device_count) {
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

typedef enum aclrtMemLocationType {
    ACL_MEM_LOCATION_TYPE_HOST = 0, /**< reserved enum, current version not support */
    ACL_MEM_LOCATION_TYPE_DEVICE,
    ACL_MEM_LOCATION_TYPE_UNREGISTERED,
    ACL_MEM_LOCATION_TYPE_HOST_NUMA = 4, /*alloc host memeory via NUMA ID */
} aclrtMemLocationType;

typedef struct aclrtMemLocation {
    uint32_t id;
    aclrtMemLocationType type;
} aclrtMemLocation;

typedef struct aclrtPtrAttributes {
    aclrtMemLocation location;
    uint32_t pageSize;
    uint32_t rsv[4];
} aclrtPtrAttributes;

ACL_FUNC_VISIBILITY aclError aclrtPointerGetAttributes(const void *ptr,
                                                       aclrtPtrAttributes *attributes) {
    std::ostringstream log;
    log << "[aclrtPointerGetAttributes] ptr=" << ptr;

    if (!ptr || !attributes) {
        log << "\n    invalid argument → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    // В эмуляторе вся память считается устройством
    attributes->location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    attributes->location.id = 0;

    log << "\n    type=DEVICE";
    log_output(log);
    return ACL_SUCCESS;
}


typedef enum {
    ACL_RT_IPC_MEM_ATTR_ACCESS_LINK,
} aclrtIpcMemAttrType;

typedef struct {
    uint32_t sdid;  // whitelisted 
    int32_t *pid;
    size_t num;
} aclrtServerPid;

// Вспомогательная структура для хранения информации об экспортированной памяти
struct IpcMemoryEntry {
    void* devPtr;           // оригинальный указатель (для локального доступа)
    size_t size;
    int shm_fd;             // файловый дескриптор shared memory
    void* mapped_addr;      // адрес mmap в экспортирующем процессе (может != devPtr)
};

static std::unordered_map<std::string, IpcMemoryEntry> g_ipc_memory_map;
static std::mutex g_ipc_mutex;

static std::string make_shm_name(const std::string& key) {
    // POSIX shm_open требует имя, начинающееся с '/'
    return "/" + key;
}

ACL_FUNC_VISIBILITY aclError aclrtIpcMemGetExportKey(
    void *devPtr, size_t size, char *key, size_t len, uint64_t flags) {
    std::ostringstream log;
    log << "[aclrtIpcMemGetExportKey] devPtr=" << devPtr
        << " size=" << size << " key=" << static_cast<void*>(key)
        << " len=" << len << " flags=" << flags;

    if (!devPtr || !key || len == 0) {
        log << "\n    invalid param → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    // Генерируем ключ
    std::ostringstream key_stream;
    key_stream << "not_npu_ipc_" << devPtr << "_" << size << "_" << flags;
    std::string keystr = key_stream.str();
    if (keystr.size() + 1 > len) {
        log << "\n    key buffer too small → ACL_ERROR_FAILURE";
        log_output(log, true);
        return ACL_ERROR_FAILURE;
    }

    // Копируем ключ наружу
    std::memcpy(key, keystr.c_str(), keystr.size() + 1);
    log << "\n    generated key=\"" << keystr << "\"";

    // Создаём POSIX shared memory
    std::string shm_name = make_shm_name(keystr);
    int shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0600);
    if (shm_fd < 0) {
        log << "\n    shm_open failed → ACL_ERROR_FAILURE";
        log_output(log, true);
        return ACL_ERROR_FAILURE;
    }

    // Размер: 8 байт для size_t (размер данных) + сами данные
    size_t total_size = sizeof(size_t) + size;
    if (ftruncate(shm_fd, total_size) != 0) {
        log << "\n    ftruncate failed → ACL_ERROR_FAILURE";
        close(shm_fd);
        shm_unlink(shm_name.c_str());
        log_output(log, true);
        return ACL_ERROR_FAILURE;
    }

    // mmap
    void* mapped = mmap(nullptr, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (mapped == MAP_FAILED) {
        log << "\n    mmap failed → ACL_ERROR_FAILURE";
        close(shm_fd);
        shm_unlink(shm_name.c_str());
        log_output(log, true);
        return ACL_ERROR_FAILURE;
    }

    // Записываем размер и данные
    memcpy(mapped, &size, sizeof(size_t));
    memcpy((char*)mapped + sizeof(size_t), devPtr, size);

    // Сохраняем в локальной мапе для этого процесса (на случай импорта в том же процессе)
    {
        std::lock_guard<std::mutex> lock(g_ipc_mutex);
        g_ipc_memory_map[keystr] = {devPtr, size, shm_fd, mapped};
    }

    log << "\n    shared memory created, fd=" << shm_fd << " mapped=" << mapped
        << " → ACL_SUCCESS";
    log_output(log, true);
    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtIpcMemClose(const char *key) {
    std::ostringstream log;
    log << "[aclrtIpcMemClose] key=" << (key ? key : "null");

    if (!key) {
        log << "\n    key is null → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    std::string keystr(key);
    std::string shm_name = make_shm_name(keystr);

    std::lock_guard<std::mutex> lock(g_ipc_mutex);
    auto it = g_ipc_memory_map.find(keystr);
    if (it != g_ipc_memory_map.end()) {
        munmap(it->second.mapped_addr, sizeof(size_t) + it->second.size);
        close(it->second.shm_fd);
        // !!! НЕ удаляем shm_unlink, чтобы другие процессы могли импортировать позже
        g_ipc_memory_map.erase(it);
        log << "\n    local mapping closed, shm preserved → ACL_SUCCESS";
    } else {
        // Даже если нет в мапе, не трогаем shm
        log << "\n    not found in local map, no action → ACL_SUCCESS";
    }

    log_output(log, true);
    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtIpcMemImportByKey(void **devPtr, const char *key, uint64_t flags) {
    std::ostringstream log;
    log << "[aclrtIpcMemImportByKey] devPtr=" << devPtr
        << " key=" << (key ? key : "null") << " flags=" << flags;

    if (!devPtr || !key) {
        log << "\n    invalid param → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return ACL_ERROR_INVALID_PARAM;
    }

    std::string keystr(key);
    std::string shm_name = make_shm_name(keystr);

    // Сначала проверяем локальную мапу (вдруг в том же процессе)
    {
        std::lock_guard<std::mutex> lock(g_ipc_mutex);
        auto it = g_ipc_memory_map.find(keystr);
        if (it != g_ipc_memory_map.end()) {
            *devPtr = it->second.devPtr;
            log << "\n    found in local map, devPtr=" << *devPtr << " → ACL_SUCCESS";
            log_output(log, true);
            return ACL_SUCCESS;
        }
    }

    // Не в локальной мапе — открываем shared memory
    int shm_fd = shm_open(shm_name.c_str(), O_RDONLY, 0);
    if (shm_fd < 0) {
        log << "\n    shm_open failed, key not exported → ACL_ERROR_FAILURE";
        log_output(log, true);
        return ACL_ERROR_FAILURE;
    }

    // Узнаем размер
    struct stat st;
    if (fstat(shm_fd, &st) != 0 || st.st_size < (off_t)sizeof(size_t)) {
        log << "\n    fstat failed or too small → ACL_ERROR_FAILURE";
        close(shm_fd);
        log_output(log, true);
        return ACL_ERROR_FAILURE;
    }

    // mmap
    void* mapped = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, shm_fd, 0);
    if (mapped == MAP_FAILED) {
        log << "\n    mmap failed → ACL_ERROR_FAILURE";
        close(shm_fd);
        log_output(log, true);
        return ACL_ERROR_FAILURE;
    }

    // Читаем размер из начала
    size_t data_size;
    memcpy(&data_size, mapped, sizeof(size_t));
    void* data_ptr = (char*)mapped + sizeof(size_t);

    // Сохраняем в локальной мапе для последующих импортов
    {
        std::lock_guard<std::mutex> lock(g_ipc_mutex);
        g_ipc_memory_map[keystr] = {data_ptr, data_size, shm_fd, mapped};
    }

    *devPtr = data_ptr;
    log << "\n    imported via shared memory, devPtr=" << data_ptr << " size=" << data_size
        << " → ACL_SUCCESS";
    log_output(log, true);
    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtIpcMemSetImportPid(const char *key, int32_t *pid, size_t num) {
    std::ostringstream log;
    log << "[aclrtIpcMemSetImportPid] key=" << (key ? key : "null")
        << " pid=" << static_cast<void*>(pid) << " num=" << num;
    // Заглушка
    log << "\n    ignored → ACL_SUCCESS";
    log_output(log, true);
    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtIpcMemSetAttr(const char *key, aclrtIpcMemAttrType type, uint64_t attr) {
    std::ostringstream log;
    log << "[aclrtIpcMemSetAttr] key=" << (key ? key : "null")
        << " type=" << type << " attr=" << attr;
    log << "\n    ignored → ACL_SUCCESS";
    log_output(log, true);
    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtIpcMemImportPidInterServer(const char *key, aclrtServerPid *serverPids, size_t num) {
    std::ostringstream log;
    log << "[aclrtIpcMemImportPidInterServer] key=" << (key ? key : "null")
        << " serverPids=" << static_cast<void*>(serverPids) << " num=" << num;
    log << "\n    ignored → ACL_SUCCESS";
    log_output(log, true);
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
    log_output(log, true);

    return h;
}

MSVP_PROF_API int32_t MsprofReportApi(uint32_t nonPersistantFlag, const struct MsprofApi *api) {
    std::ostringstream log;
    log << "[MsprofReportApi] nonPersistantFlag=" << nonPersistantFlag
        << " api=" << api;

    if (!api) {
        log << "\n    api is null → FAILED";
        log_output(log, true);
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
    log_output(log, true);

    // not‑NPU: no-op
    return 0;
}

MSVP_PROF_API int32_t MsprofReportCompactInfo(uint32_t nonPersistantFlag, const VOID_PTR data, uint32_t length) {
    std::ostringstream log;
    log << "[MsprofReportCompactInfo] nonPersistantFlag=" << nonPersistantFlag
        << " length=" << length
        << " data=" << data;

    if (!data) {
        log << "\n    data is null → FAILED";
        log_output(log, true);
        return -1;
    }

    if (length < sizeof(MsprofCompactInfo)) {
        log << "\n    length < sizeof(MsprofCompactInfo) → FAILED";
        log_output(log, true);
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
    log_output(log, true);

    // not‑NPU: no-op
    return 0;
}

MSVP_PROF_API uint64_t MsprofSysCycleTime(void) {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    uint64_t time = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();

    std::ostringstream oss;
    oss << "[MsprofSysCycleTime] time=" << time;
    log_output(oss, true);

    return time;
}

MSVP_PROF_API int32_t MsprofReportAdditionalInfo(uint32_t nonPersistantFlag, const VOID_PTR data, uint32_t length) {
    std::ostringstream log;
    log << "[MsprofReportAdditionalInfo] nonPersistantFlag=" << nonPersistantFlag
        << " data=" << data << " length=" << length;
    log_output(log, true);

    return 0; // SUCCESS
}

MSVP_PROF_API int32_t MsprofRegisterCallback(uint32_t moduleId, ProfCommandHandle handle) {
    std::ostringstream log;
    log << "[MsprofRegisterCallback] moduleId=" << moduleId
        << " handle=" << reinterpret_cast<const void*>(handle);
    log_output(log);

    return 0; // SUCCESS
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

    attr->values[attrName] = attrValue;

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

    attr->values[attrName] = attrValue;

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

    attr->values[attrName] = attrValue;

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

    attr->values[attrName] = attrValue;

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

    attr->values[attrName] = std::vector<uint8_t>(values, values + numValues);

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

    attr->values[attrName] = std::vector<float>(values, values + numValues);

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

    attr->values[attrName] = std::vector<int64_t>(values, values + numValues);

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

    for (int i = 0; i < numLists; ++i) {
        int n = numValues[i];
        const int64_t* src = values[i];
        if (!src || n < 0) {
            log << "\n    invalid inner list → ACL_ERROR_INVALID_PARAM";
            log_output(log, true);
            return ACL_ERROR_INVALID_PARAM;
        }
        out.emplace_back(src, values[i] + n);
    }

    attr->values[attrName] = std::move(out);

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

    attr->values[attrName] = std::string(attrValue);

 // log << "\n    stored string: " << attrName
 //     << "=\"" << attrValue << "\"";
 // log_output(log);

    return ACL_SUCCESS;
}



#ifdef __cplusplus
}
#endif

#endif // NOT_ACL
