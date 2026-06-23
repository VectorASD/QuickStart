#include "common.h"  // log_output
#include "../env/include/runtime/runtime/rt.h"

#include <cstring>   // std::strncpy

void __not_runtime_placeholder() {}



static uint32_t g_device_count = 1;
static uint32_t g_current_device = 0;

static const char* g_device_name = "Ascend950PR_957c";

static int64_t info_ai_core_num     = 28;


extern "C" {

RTS_API rtError_t rtGetDeviceCount(int32_t *count) {
    std::ostringstream log;
    log << "[rtGetDeviceCount]";
    if (!count) {
        log << "\n    count is null → RT_ERROR_INVALID_VALUE";
        log_output(log, true);
        return RT_ERROR_INVALID_VALUE;
    }

    *count = g_device_count;
    log << " count=" << *count;
    log_output(log);

    return RT_ERROR_NONE;
}

RTS_API rtError_t rtGetDevice(int32_t *deviceId) {
    std::ostringstream log;
    log << "[rtGetDevice]";
    if (!deviceId) {
        log << "\n    deviceId is null → RT_ERROR_INVALID_VALUE";
        log_output(log, true);
        return RT_ERROR_INVALID_VALUE;
    }

    *deviceId = g_current_device;
    log << " device=" << *deviceId;
    log_output(log);

    return RT_ERROR_NONE;
}

RTS_API rtError_t rtSetDevice(int32_t deviceId) {
    std::ostringstream log;
    log << "[rtSetDevice] device=" << deviceId;
    if (deviceId < 0 || deviceId >= g_device_count) {
        log << "\n    invalid deviceId → ACL_ERROR_INVALID_PARAM";
        log_output(log, true);
        return RT_ERROR_INVALID_VALUE;
    }

    g_current_device = deviceId;
    log_output(log);

    return RT_ERROR_NONE;
}

rtError_t rtGetSocVersion(char_t *ver, const uint32_t maxLen) {
    std::strncpy(ver, g_device_name, maxLen - 1);
    ver[maxLen - 1] = '\0';

    std::ostringstream log;
    log << "[rtGetSocVersion] returned='" << ver << '\'';
    log_output(log);

    return RT_ERROR_NONE;
}


RTS_API rtError_t rtGetAiCoreCount(uint32_t *aiCoreCnt) {
    std::ostringstream log;
    log << "[rtGetAiCoreCount]";
    if (!aiCoreCnt) {
        log << "\n    aiCoreCnt is null → RT_ERROR_INVALID_VALUE";
        log_output(log, true);
        return RT_ERROR_INVALID_VALUE;
    }

    *aiCoreCnt = info_ai_core_num;
    log << " aiCoreCnt=" << *aiCoreCnt;
    log_output(log);

    return RT_ERROR_NONE;
}



RTS_API rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **hdl) {
    std::ostringstream log;
    log << "[rtDevBinaryRegister] bin=" << bin
        << " hdl=" << hdl;

    if (!bin || !hdl) {
        log << "\n    bin or hdl is null → RT_ERROR_INVALID_VALUE";
        log_output(log, true);
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

    log << "\n    handle NOT set to bin";
    log_output(log, true);

    return RT_ERROR_NONE;
}

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
        log_output(log, true);
        return RT_ERROR_INVALID_VALUE;
    }
    log_output(log, true);

    // not‑NPU: Torch-NPU не использует реальную регистрацию функций.
    // Мы просто логируем и возвращаем успех.

    return RT_ERROR_NONE;
}


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
        log_output(log, true);
        return RT_ERROR_INVALID_VALUE;
    }

    // Логируем smDesc, если он есть
    if (smDesc) {
        log << "\n    rtSmDesc_t:"
            << "\n        size=" << smDesc->size
            << "\n        l2_in_main=" << (int)smDesc->l2_in_main

            << "\n        remap[0..7]: ";
        for (int i = 0; i < 8; i++)
            log << (int) smDesc->remap[i] << " ";

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

    log << "\n    kernel NOT launched";
    log_output(log, true);

    return RT_ERROR_NONE;
}

RTS_API rtError_t rtKernelLaunchWithFlag(const void *stubFunc, uint32_t numBlocks,
                                         rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc,
                                         rtStream_t stm, uint32_t flags) {
    return rtKernelLaunchWithFlagV2(stubFunc, numBlocks, argsInfo, smDesc, stm, flags, nullptr);
}

RTS_API rtError_t rtKernelLaunchWithFlagV2(const void *stubFunc, uint32_t numBlocks,
                                           rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc,
                                           rtStream_t stm, uint32_t flags,
                                           const rtTaskCfgInfo_t *cfgInfo) {
    std::ostringstream log;
    log << "[rtKernelLaunchWithFlagV2] stubFunc=" << stubFunc
        << " numBlocks=" << numBlocks
        << " argsInfo=" << argsInfo
        << " smDesc=" << smDesc
        << " stream=" << stm
        << " flags=" << flags
        << " cfgInfo=" << cfgInfo;

    if (!stubFunc) {
        log << "\n    stubFunc is null → RT_ERROR_INVALID_VALUE";
        log_output(log, true);
        return RT_ERROR_INVALID_VALUE;
    }

    // Логируем argsInfo
    if (argsInfo) {
        log << "\n    rtArgsEx_t:"
            << "\n        args=" << argsInfo->args
            << " argsSize=" << argsInfo->argsSize
            << " tilingAddrOffset=" << argsInfo->tilingAddrOffset
            << " tilingDataOffset=" << argsInfo->tilingDataOffset
            << " hostInputInfoNum=" << argsInfo->hostInputInfoNum
            << " hasTiling=" << (int)argsInfo->hasTiling
            << " isNoNeedH2DCopy=" << (int)argsInfo->isNoNeedH2DCopy;
        if (argsInfo->hostInputInfoPtr && argsInfo->hostInputInfoNum > 0) {
            for (uint16_t i = 0; i < argsInfo->hostInputInfoNum; ++i) {
                const auto &hi = argsInfo->hostInputInfoPtr[i];
                log << "\n        hostInputInfo[" << i << "]: addrOffset=" << hi.addrOffset
                    << " dataOffset=" << hi.dataOffset;
            }
        }
    }

    // Логируем smDesc (как в rtKernelLaunch)
    if (smDesc) {
        log << "\n    rtSmDesc_t:"
            << "\n        size=" << smDesc->size
            << "\n        l2_in_main=" << (int)smDesc->l2_in_main;

        log << "\n        remap[0..7]: ";
        for (int i = 0; i < 8; i++)
            log << (int)smDesc->remap[i] << " ";

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

    // Логируем cfgInfo
    if (cfgInfo) {
        log << "\n    rtTaskCfgInfo_t:"
            << "\n        qos=" << (int)cfgInfo->qos
            << " partId=" << (int)cfgInfo->partId
            << " schemMode=" << (int)cfgInfo->schemMode
            << " d2dCrossFlag=" << cfgInfo->d2dCrossFlag
            << " blockDimOffset=" << cfgInfo->blockDimOffset
            << " dumpflag=" << (int)cfgInfo->dumpflag
            << " neverTimeout=" << (int)cfgInfo->neverTimeout
            << " localMemorySize=" << cfgInfo->localMemorySize;
    }

    log << "\n    kernel NOT launched";
    log_output(log, true);

    return RT_ERROR_NONE;
}



RTS_API rtError_t rtStreamCreate(rtStream_t *stream, int32_t priority) {
    std::ostringstream log;
    log << "[rtStreamCreate] priority=" << priority;

    if (!stream) {
        log << "\n    stream_ptr is null → RT_ERROR_INVALID_VALUE";
        log_output(log, true);
        return RT_ERROR_INVALID_VALUE;
    }

    *stream = malloc(1);
    log << "\n    created stream=" << stream;
    log_output(log);

    return RT_ERROR_NONE;
}
RTS_API rtError_t rtStreamDestroy(rtStream_t stream) {
    std::ostringstream log;
    log << "[rtStreamDestroy] stream=" << stream;
    log_output(log);

    free(stream);

    return ACL_SUCCESS;
}

RTS_API rtError_t rtMallocHost(void **hostPtr, uint64_t size, const uint16_t moduleId) {
    std::ostringstream log;
    log << "[rtMallocHost] size=" << size << " moduleId=" << moduleId;

    if (!hostPtr) {
        log << "\n    hostPtr is null → RT_ERROR_INVALID_VALUE";
        log_output(log, true);
        return RT_ERROR_INVALID_VALUE;
    }

    void *ptr = malloc(size);
    if (!ptr) {
        log << "\n    allocation failed → ACL_ERROR_BAD_ALLOC";
        log_output(log, true);
        return RT_ERROR_BAD_ALLOC;
    }

    *hostPtr = ptr;
    log << "\n    allocated=" << ptr;
    log_output(log);

    return RT_ERROR_NONE;
}
RTS_API rtError_t rtFreeHost(void *hostPtr) {
    std::ostringstream log;
    log << "[rtFreeHost] hostPtr=" << hostPtr;
    if (!hostPtr) {
        log << "\n    hostPtr is null → RT_ERROR_INVALID_VALUE";
        log_output(log, true);
        return RT_ERROR_INVALID_VALUE;
    }

    free(hostPtr);
    log_output(log);

    return RT_ERROR_NONE;
}

RTS_API rtError_t rtMalloc(void **devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId) {
    std::ostringstream log;
    log << "[rtMalloc] size=" << size << " type=" << type << " moduleId=" << moduleId;

    if (!devPtr) {
        log << "\n    devPtr is null → RT_ERROR_INVALID_VALUE";
        log_output(log, true);
        return RT_ERROR_INVALID_VALUE;
    }

    void *ptr = malloc(size);
    if (!ptr) {
        log << "\n    allocation failed → RT_ERROR_BAD_ALLOC";
        log_output(log, true);
        return RT_ERROR_BAD_ALLOC;
    }

    *devPtr = ptr;
    log << "\n    allocated=" << ptr;
    log_output(log);

    return RT_ERROR_NONE;
}
RTS_API rtError_t rtFree(void *devPtr) {
    std::ostringstream log;
    log << "[rtFree] devPtr=" << devPtr;
    if (!devPtr) {
        log << "\n    devPtr is null → RT_ERROR_INVALID_VALUE";
        log_output(log, true);
        return RT_ERROR_INVALID_VALUE;
    }

    free(devPtr);
    log_output(log);

    return RT_ERROR_NONE;
}
RTS_API rtError_t rtMemcpy(void *dst, uint64_t destMax, const void *src, uint64_t cnt, rtMemcpyKind_t kind) {
    std::ostringstream log;
    log << "[rtMemcpy] dst=" << dst << " src=" << src << " cnt=" << cnt << " destMax=" << destMax << " kind=" << kind;

    if (!dst || !src) {
        log << "\n    null pointer → RT_ERROR_INVALID_VALUE";
        log_output(log, true);
        return RT_ERROR_INVALID_VALUE;
    }
    if (cnt > destMax) {
        log << "\n    cnt > destMax → RT_ERROR_INVALID_VALUE";
        log_output(log, true);
        return RT_ERROR_INVALID_VALUE;
    }

    std::memcpy(dst, src, cnt);
    log_output(log);

    return RT_ERROR_NONE;
}

}
