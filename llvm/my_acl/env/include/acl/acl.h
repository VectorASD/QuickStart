#include <cstddef>   // size_t
#include <stdint.h>  // int32_t, uint16_t, uint32_t, uint64_t, uint8_t


#ifndef ACL_H
#define ACL_H


// ~~~ cann-npu-runtime/runtime/pkg_inc/profiling/aprof_pub.h ~~~

#define MSPROF_REPORT_DATA_MAGIC_NUM 0x5A5AU
#define PATH_LEN_MAX 1023
#define PARAM_LEN_MAX 4095
#define MSPROF_MAX_DEV_NUM 64
#define MSPROF_COMPACT_INFO_DATA_LENGTH 40

#define MSPROF_REPORT_NODE_LEVEL        10000U
#define MSPROF_REPORT_NODE_BASIC_INFO_TYPE       0U  /* type info: node_basic_info */
#define MSPROF_REPORT_NODE_TENSOR_INFO_TYPE      1U  /* type info: tensor_info */
#define MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE  4U  /* type info: context_id_info */
#define MSPROF_REPORT_NODE_LAUNCH_TYPE           5U  /* type info: launch */

#define MSPROF_GE_TENSOR_DATA_SHAPE_LEN 8
#define MSPROF_GE_TENSOR_DATA_NUM 5
#define MSPROF_CTX_ID_MAX_NUM 55


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

#define MSPROF_ADDTIONAL_INFO_DATA_LENGTH (232)
struct MsprofAdditionalInfo {  // for MsprofReportAdditionalInfo buffer data
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
    uint8_t  data[MSPROF_ADDTIONAL_INFO_DATA_LENGTH];
};

struct MsrofTensorData {
    uint32_t tensorType;
    uint32_t format;
    uint32_t dataType;
    uint32_t shape[MSPROF_GE_TENSOR_DATA_SHAPE_LEN];
};

struct MsprofTensorInfo {
    uint64_t opName;
    uint32_t tensorNum;
    struct MsrofTensorData tensorData[MSPROF_GE_TENSOR_DATA_NUM];
};

struct MsprofContextIdInfo {
    uint64_t opName;
    uint32_t ctxIdNum;
    uint32_t ctxIds[MSPROF_CTX_ID_MAX_NUM];
};


enum MsprofGeTaskType { 
    MSPROF_GE_TASK_TYPE_AI_CORE = 0,
    MSPROF_GE_TASK_TYPE_AI_CPU,
    MSPROF_GE_TASK_TYPE_AIV,
    MSPROF_GE_TASK_TYPE_WRITE_BACK,
    MSPROF_GE_TASK_TYPE_MIX_AIC,
    MSPROF_GE_TASK_TYPE_MIX_AIV,
    MSPROF_GE_TASK_TYPE_FFTS_PLUS,
    MSPROF_GE_TASK_TYPE_DSA,
    MSPROF_GE_TASK_TYPE_DVPP,
    MSPROF_GE_TASK_TYPE_HCCL,
    MSPROF_GE_TASK_TYPE_FUSION,
    MSPROF_GE_TASK_TYPE_INVALID
};

enum MsprofGeTensorType {
    MSPROF_GE_TENSOR_TYPE_INPUT = 0,
    MSPROF_GE_TENSOR_TYPE_OUTPUT,
};


typedef int32_t (*ProfCommandHandle)(uint32_t type, void *data, uint32_t len);


#if (defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER))
#define MSVP_PROF_API __declspec(dllexport)
#else
#define MSVP_PROF_API __attribute__((visibility("default")))
#endif

typedef void* VOID_PTR;

#ifdef __cplusplus
extern "C" {
#endif

MSVP_PROF_API uint64_t MsprofGetHashId(const char *hashInfo, size_t length);
MSVP_PROF_API int32_t MsprofReportApi(uint32_t nonPersistantFlag, const struct MsprofApi *api);
MSVP_PROF_API int32_t MsprofReportCompactInfo(uint32_t nonPersistantFlag, const VOID_PTR data, uint32_t length);
MSVP_PROF_API uint64_t MsprofSysCycleTime(void);
MSVP_PROF_API int32_t MsprofReportAdditionalInfo(uint32_t nonPersistantFlag, const VOID_PTR data, uint32_t length);
MSVP_PROF_API int32_t MsprofRegisterCallback(uint32_t moduleId, ProfCommandHandle handle);

#ifdef __cplusplus
}
#endif


#endif  // ACL_H
