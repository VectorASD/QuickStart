#include <stdint.h>  // int32_t, uint32_t, uint64_t


#ifndef RT_H
#define RT_H

typedef void *rtStream_t;
typedef uint32_t rtMemType_t;
typedef int32_t rtError_t;

static const int32_t RT_ERROR_NONE = 0; // success
static const int32_t RT_ERROR_INVALID_VALUE = 1; // ???
static const int32_t RT_ERROR_BAD_ALLOC = 2;

#ifndef char_t
    typedef char char_t;
#endif

#define RT_DEV_BINARY_MAGIC_ELF 0x43554245U
#define RT_DEV_BINARY_MAGIC_ELF_AICPU 0x41415243U
#define RT_DEV_BINARY_MAGIC_ELF_AIVEC 0x41415246U
#define RT_DEV_BINARY_MAGIC_ELF_AICUBE 0x41494343U

#define RT_MEMORY_HBM (0x2U)       // HBM memory on device
#define RT_MEMORY_HOST (0x81U)     // Memory on host


#ifdef __cplusplus
extern "C" {
#endif

typedef struct tagRtDevBinary {
    uint32_t magic;    // magic number
    uint32_t version;  // version of binary
    const void *data;  // binary data
    uint64_t length;   // binary length
} rtDevBinary_t;

typedef enum tagRtMemcpyKind {
    RT_MEMCPY_HOST_TO_HOST = 0,  // host to host
    RT_MEMCPY_HOST_TO_DEVICE,    // host to device
    RT_MEMCPY_DEVICE_TO_HOST,    // device to host
    RT_MEMCPY_DEVICE_TO_DEVICE,  // device to device, 1P && P2P
    RT_MEMCPY_MANAGED,           // managed memory
    RT_MEMCPY_ADDR_DEVICE_TO_DEVICE,
    RT_MEMCPY_HOST_TO_DEVICE_EX, // host  to device ex (only used for 8 bytes)
    RT_MEMCPY_DEVICE_TO_HOST_EX, // device to host ex
    RT_MEMCPY_DEFAULT,           // auto infer copy dir
    RT_MEMCPY_RESERVED,
} rtMemcpyKind_t;

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

typedef struct rtHostInputInfo {
    uint32_t addrOffset;
    uint32_t dataOffset;
} rtHostInputInfo_t;

typedef struct tagRtArgsEx {
    void *args;                     // args host mem addr
    rtHostInputInfo_t *hostInputInfoPtr;     // nullptr means no host mem input
    uint32_t argsSize;              // input + output + tiling addr size + tiling data size + host mem
    uint32_t tilingAddrOffset;      // tiling addr offset
    uint32_t tilingDataOffset;      // tiling data offset
    uint16_t hostInputInfoNum;      // hostInputInfo num
    uint8_t hasTiling;              // if has tiling: 0 means no tiling
    uint8_t isNoNeedH2DCopy;        // is no need host to device copy: 0 means need H2D copy,
                                    // others means doesn't need H2D copy.
    uint8_t reserved[4];
} rtArgsEx_t;

typedef struct tagRtTaskCfgInfo {
    uint8_t qos;
    uint8_t partId;
    uint8_t schemMode; // rtschemModeType_t 0:normal;1:batch;2:sync
    bool d2dCrossFlag; // d2dCrossFlag true:D2D_CROSS flase:D2D_INNER
    uint32_t blockDimOffset;
    uint8_t dumpflag; // dumpflag 0:fault 2:RT_KERNEL_DUMPFLAG 4:RT_FUSION_KERNEL_DUMPFLAG
    uint8_t neverTimeout; // 1: never timeout, 0: will timeout
    uint8_t rev[2];
    uint32_t localMemorySize;  // for simt ub_size
} rtTaskCfgInfo_t;


#define RTS_DLL_EXPORT

#ifndef RTS_API
#ifdef RTS_DLL_EXPORT
#define RTS_API __attribute__((visibility("default")))
#else
#define RTS_API
#endif
#endif

RTS_API rtError_t rtGetDeviceCount(int32_t *cnt);
RTS_API rtError_t rtGetDevice(int32_t *devId);
RTS_API rtError_t rtSetDevice(int32_t devId);
RTS_API rtError_t rtGetSocVersion(char_t *ver, const uint32_t maxLen);

RTS_API rtError_t rtGetAiCoreCount(uint32_t *aiCoreCnt);

/**
 * @ingroup rt_kernel
 * @brief register device binary
 * @param [in] bin   device binary description
 * @param [out] hdl   device binary handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **hdl);

/**
 * @ingroup rt_kernel
 * @brief register device function
 * @param [in] binHandle   device binary handle
 * @param [in] stubFunc   stub function
 * @param [in] stubName   stub function name
 * @param [in] kernelInfoExt   kernel Info extension. device function description or tiling key,
 *                             depending static shape or dynmaic shape.
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtFunctionRegister(void *binHandle, const void *stubFunc, const char_t *stubName,
                                     const void *kernelInfoExt, uint32_t funcMode);

/**
 * @ingroup rt_kernel
 * @brief launch kernel to device
 * @param [in] stubFunc   stub function
 * @param [in] numBlocks   block dimensions
 * @param [in] args   argments address for kernel function
 * @param [in] argsSize   argements size
 * @param [in] smDesc   shared memory description
 * @param [in] stm   associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelLaunch(const void *stubFunc, uint32_t numBlocks, void *args, uint32_t argsSize,
                                 rtSmDesc_t *smDesc, rtStream_t stm);

/**
 * @ingroup rtKernelLaunchWithFlag
 * @brief launch kernel to device
 * @param [in] stubFunc   stub function
 * @param [in] numBlocks   block dimensions
 * @param [in] argsInfo   argments address for kernel function
 * @param [in] smDesc     shared memory description
 * @param [in] stm        associated stream
 * @param [in] flags      dump flag
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelLaunchWithFlag(const void *stubFunc, uint32_t numBlocks, rtArgsEx_t *argsInfo,
                                         rtSmDesc_t *smDesc, rtStream_t stm, uint32_t flags);

/**
 * @ingroup rtKernelLaunchWithFlag
 * @brief launch kernel to device
 * @param [in] stubFunc   stub function
 * @param [in] numBlocks   block dimensions
 * @param [in] argsInfo   argments address for kernel function
 * @param [in] smDesc     shared memory description
 * @param [in] stm        associated stream
 * @param [in] flags      dump flag
 * @param [in] cfgInfo      task config info
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelLaunchWithFlagV2(const void *stubFunc, uint32_t numBlocks, rtArgsEx_t *argsInfo,
    rtSmDesc_t *smDesc, rtStream_t stm, uint32_t flags, const rtTaskCfgInfo_t *cfgInfo);


RTS_API rtError_t rtStreamCreate(rtStream_t *stm, int32_t priority);
RTS_API rtError_t rtStreamDestroy(rtStream_t stm);

RTS_API rtError_t rtMallocHost(void **hostPtr, uint64_t size, const uint16_t moduleId);
RTS_API rtError_t rtFreeHost(void *hostPtr);

RTS_API rtError_t rtMalloc(void **devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId);
RTS_API rtError_t rtFree(void *devPtr);

RTS_API rtError_t rtMemcpy(void *dst, uint64_t destMax, const void *src, uint64_t cnt, rtMemcpyKind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // RT_H
