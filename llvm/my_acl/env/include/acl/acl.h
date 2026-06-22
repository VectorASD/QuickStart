#include <stdint.h>  // uint16_t, uint32_t, uint64_t
#include <thread>    // get_id, thread
#include "../../../src/common.h"


#ifndef ACL_H
#define ACL_H


// ~~~ cann-npu-runtime/runtime/pkg_inc/profiling/aprof_pub.h ~~~

#define MSPROF_REPORT_DATA_MAGIC_NUM 0x5A5AU
#define PATH_LEN_MAX 1023
#define PARAM_LEN_MAX 4095
#define MSPROF_MAX_DEV_NUM 64
#define MSPROF_COMPACT_INFO_DATA_LENGTH 40

#if (defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER))
#define MSVP_PROF_API __declspec(dllexport)
#else
#define MSVP_PROF_API __attribute__((visibility("default")))
#endif


struct MsprofCommandHandleParams {
    uint32_t pathLen;
    uint32_t storageLimit;  // MB
    uint32_t profDataLen;
    char path[PATH_LEN_MAX + 1];
    char profData[PARAM_LEN_MAX + 1];
};

struct MsprofCommandHandle {
    uint64_t profSwitch;
    uint64_t profSwitchHi;
    uint32_t devNums;
    uint32_t devIdList[MSPROF_MAX_DEV_NUM];
    uint32_t modelId;
    uint32_t type;
    uint32_t cacheFlag;
    struct MsprofCommandHandleParams params;
};

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


#endif  // ACL_H
