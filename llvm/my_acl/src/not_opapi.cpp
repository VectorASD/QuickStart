// ===================================================================
//  not_opapi.cpp – собственная реализация API libopapi (aclnn)
//  в составе проекта not_npu.
//
//  Часть операций Ascend CANN (например, torch.angle) использует
//  не стандартный aclopCompileAndExecute(V2), а отдельное API,
//  экспортируемое библиотекой libopapi.so.  Заголовки этого API
//  находятся в пакете: cann-opbase.run/ops_base/aclnn/acl_meta.h
// ===================================================================

#include "common.h"     // log_output, ...
#include "helpers.cpp"
#include <cstdint>      // int32_t, int64_t, uint64_t

#ifdef __cplusplus
extern "C" {
#endif

struct aclOpExecutor {};
struct aclTensor {};
struct aclScalar {};
struct aclIntArray {};
struct aclFloatArray {};
struct aclBoolArray {};
struct aclTensorList {};
struct aclScalarList {};

typedef int32_t aclnnStatus;
constexpr aclnnStatus OK = 0;
constexpr aclnnStatus UNIMPLEMENTED = 1;



// ~~~ Основное meta-API ~~~

aclTensor* aclCreateTensor(const int64_t* viewDims, uint64_t viewDimsNum, aclDataType dataType,
                           const int64_t* stride, int64_t offset, aclFormat format,
                           const int64_t* storageDims, uint64_t storageDimsNum,
                           void* tensorData) {
    std::ostringstream log;
    log << "[aclCreateTensor] viewDimsNum=" << viewDimsNum
        << " dataType=" << dataType
        << " tensorData=" << tensorData;
    log_output(log, true);
    return nullptr;   // заглушка
}

aclScalar* aclCreateScalar(void* value, aclDataType dataType) {
    std::ostringstream log;
    log << "[aclCreateScalar] value=" << value
        << " dataType=" << dataType;
    log_output(log, true);
    return nullptr;   // заглушка
}

aclIntArray* aclCreateIntArray(const int64_t* value, uint64_t size) {
    std::ostringstream log;
    log << "[aclCreateIntArray] value=" << static_cast<const void*>(value)
        << " size=" << size;
    log_output(log, true);
    return nullptr;
}

aclFloatArray* aclCreateFloatArray(const float* value, uint64_t size) {
    std::ostringstream log;
    log << "[aclCreateFloatArray] value=" << static_cast<const void*>(value)
        << " size=" << size;
    log_output(log, true);
    return nullptr;
}

aclBoolArray* aclCreateBoolArray(const bool* value, uint64_t size) {
    std::ostringstream log;
    log << "[aclCreateBoolArray] value=" << static_cast<const void*>(value)
        << " size=" << size;
    log_output(log, true);
    return nullptr;
}

aclTensorList* aclCreateTensorList(const aclTensor* const* value, uint64_t size) {
    std::ostringstream log;
    log << "[aclCreateTensorList] value=" << static_cast<const void*>(value)
        << " size=" << size;
    log_output(log, true);
    return nullptr;
}

aclScalarList* aclCreateScalarList(const aclScalar* const* value, uint64_t size) {
    std::ostringstream log;
    log << "[aclCreateScalarList] value=" << static_cast<const void*>(value)
        << " size=" << size;
    log_output(log, true);
    return nullptr;
}

aclnnStatus aclDestroyTensor(const aclTensor* tensor) {
    std::ostringstream log;
    log << "[aclDestroyTensor] tensor=" << static_cast<const void*>(tensor);
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclDestroyScalar(const aclScalar* scalar) {
    std::ostringstream log;
    log << "[aclDestroyScalar] scalar=" << static_cast<const void*>(scalar);
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclDestroyIntArray(const aclIntArray* array) {
    std::ostringstream log;
    log << "[aclDestroyIntArray] array=" << static_cast<const void*>(array);
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclDestroyFloatArray(const aclFloatArray* array) {
    std::ostringstream log;
    log << "[aclDestroyFloatArray] array=" << static_cast<const void*>(array);
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclDestroyBoolArray(const aclBoolArray* array) {
    std::ostringstream log;
    log << "[aclDestroyBoolArray] array=" << static_cast<const void*>(array);
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclDestroyTensorList(const aclTensorList* array) {
    std::ostringstream log;
    log << "[aclDestroyTensorList] array=" << static_cast<const void*>(array);
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclDestroyScalarList(const aclScalarList* array) {
    std::ostringstream log;
    log << "[aclDestroyScalarList] array=" << static_cast<const void*>(array);
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclGetViewShape(const aclTensor* tensor, int64_t** viewDims, uint64_t* viewDimsNum) {
    std::ostringstream log;
    log << "[aclGetViewShape] tensor=" << static_cast<const void*>(tensor)
        << " viewDims=" << static_cast<void*>(viewDims)
        << " viewDimsNum=" << static_cast<void*>(viewDimsNum);
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclGetDataType(const aclTensor* tensor, aclDataType* dataType) {
    std::ostringstream log;
    log << "[aclGetDataType] tensor=" << static_cast<const void*>(tensor)
        << " dataType=" << static_cast<void*>(dataType);
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclGetStorageShape(const aclTensor* tensor, int64_t** storageDims, uint64_t* storageDimsNum) {
    std::ostringstream log;
    log << "[aclGetStorageShape] tensor=" << static_cast<const void*>(tensor)
        << " storageDims=" << static_cast<void*>(storageDims)
        << " storageDimsNum=" << static_cast<void*>(storageDimsNum);
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclGetViewStrides(const aclTensor* tensor, int64_t** stridesValue, uint64_t* stridesNum) {
    std::ostringstream log;
    log << "[aclGetViewStrides] tensor=" << static_cast<const void*>(tensor)
        << " stridesValue=" << static_cast<void*>(stridesValue)
        << " stridesNum=" << static_cast<void*>(stridesNum);
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclGetViewOffset(const aclTensor* tensor, int64_t* offset) {
    std::ostringstream log;
    log << "[aclGetViewOffset] tensor=" << static_cast<const void*>(tensor)
        << " offset=" << static_cast<void*>(offset);
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclGetFormat(const aclTensor* tensor, aclFormat* format) {
    std::ostringstream log;
    log << "[aclGetFormat] tensor=" << static_cast<const void*>(tensor)
        << " format=" << static_cast<void*>(format);
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclGetIntArraySize(const aclIntArray* array, uint64_t* size) {
    std::ostringstream log;
    log << "[aclGetIntArraySize] array=" << static_cast<const void*>(array)
        << " size=" << static_cast<void*>(size);
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclGetFloatArraySize(const aclFloatArray* array, uint64_t* size) {
    std::ostringstream log;
    log << "[aclGetFloatArraySize] array=" << static_cast<const void*>(array)
        << " size=" << static_cast<void*>(size);
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclGetBoolArraySize(const aclBoolArray* array, uint64_t* size) {
    std::ostringstream log;
    log << "[aclGetBoolArraySize] array=" << static_cast<const void*>(array)
        << " size=" << static_cast<void*>(size);
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclGetTensorListSize(const aclTensorList* tensorList, uint64_t* size) {
    std::ostringstream log;
    log << "[aclGetTensorListSize] tensorList=" << static_cast<const void*>(tensorList)
        << " size=" << static_cast<void*>(size);
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclGetScalarListSize(const aclScalarList* scalarList, uint64_t* size) {
    std::ostringstream log;
    log << "[aclGetScalarListSize] scalarList=" << static_cast<const void*>(scalarList)
        << " size=" << static_cast<void*>(size);
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclInitTensor(aclTensor* tensor, const int64_t* viewDims, uint64_t viewDimsNum,
                           aclDataType dataType, const int64_t* stride, int64_t offset,
                           aclFormat format, const int64_t* storageDims, uint64_t storageDimsNum,
                           void* tensorDataAddr) {
    std::ostringstream log;
    log << "[aclInitTensor] tensor=" << static_cast<const void*>(tensor)
        << "\n    viewDims=" << static_cast<const void*>(viewDims)
        << " viewDimsNum=" << viewDimsNum
        << "\n    dataType=" << dataType
        << " stride=" << static_cast<const void*>(stride)
        << " offset=" << offset
        << "\n    format=" << format
        << " storageDims=" << static_cast<const void*>(storageDims)
        << " storageDimsNum=" << storageDimsNum
        << " tensorDataAddr=" << tensorDataAddr;
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclSetAclOpExecutorRepeatable(aclOpExecutor* executor) {
    std::ostringstream log;
    log << "[aclSetAclOpExecutorRepeatable] executor=" << static_cast<const void*>(executor);
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclDestroyAclOpExecutor(aclOpExecutor* executor) {
    std::ostringstream log;
    log << "[aclDestroyAclOpExecutor] executor=" << static_cast<const void*>(executor);
    log_output(log, true);
    return UNIMPLEMENTED;
}

// Все функции установки адресов – одинаковые заглушки
#define IMPL_SET_TENSOR_ADDR(name) \
aclnnStatus name(aclOpExecutor* executor, const size_t index, aclTensor* tensor, void* addr) { \
    std::ostringstream log; \
    log << "[" #name "] executor=" << static_cast<const void*>(executor) \
        << " index=" << index \
        << " tensor=" << static_cast<const void*>(tensor) \
        << " addr=" << addr; \
    log_output(log, true); \
    return UNIMPLEMENTED; \
}

IMPL_SET_TENSOR_ADDR(AclSetInputTensorAddr)
IMPL_SET_TENSOR_ADDR(AclSetOutputTensorAddr)
IMPL_SET_TENSOR_ADDR(aclSetInputTensorAddr)
IMPL_SET_TENSOR_ADDR(aclSetOutputTensorAddr)
IMPL_SET_TENSOR_ADDR(AclSetTensorAddr)
IMPL_SET_TENSOR_ADDR(aclSetTensorAddr)

aclnnStatus AclSetDynamicInputTensorAddr(aclOpExecutor* executor, size_t irIndex,
                                         const size_t relativeIndex,
                                         aclTensorList* tensors, void* addr) {
    std::ostringstream log;
    log << "[AclSetDynamicInputTensorAddr] executor=" << static_cast<const void*>(executor)
        << " irIndex=" << irIndex
        << " relativeIndex=" << relativeIndex
        << " tensors=" << static_cast<const void*>(tensors)
        << " addr=" << addr;
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus AclSetDynamicOutputTensorAddr(aclOpExecutor* executor, size_t irIndex,
                                          const size_t relativeIndex,
                                          aclTensorList* tensors, void* addr) {
    std::ostringstream log;
    log << "[AclSetDynamicOutputTensorAddr] executor=" << static_cast<const void*>(executor)
        << " irIndex=" << irIndex
        << " relativeIndex=" << relativeIndex
        << " tensors=" << static_cast<const void*>(tensors)
        << " addr=" << addr;
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclSetDynamicInputTensorAddr(aclOpExecutor* executor, size_t irIndex,
                                         const size_t relativeIndex,
                                         aclTensorList* tensors, void* addr) {
    std::ostringstream log;
    log << "[aclSetDynamicInputTensorAddr] executor=" << static_cast<const void*>(executor)
        << " irIndex=" << irIndex
        << " relativeIndex=" << relativeIndex
        << " tensors=" << static_cast<const void*>(tensors)
        << " addr=" << addr;
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclSetDynamicOutputTensorAddr(aclOpExecutor* executor, size_t irIndex,
                                          const size_t relativeIndex,
                                          aclTensorList* tensors, void* addr) {
    std::ostringstream log;
    log << "[aclSetDynamicOutputTensorAddr] executor=" << static_cast<const void*>(executor)
        << " irIndex=" << irIndex
        << " relativeIndex=" << relativeIndex
        << " tensors=" << static_cast<const void*>(tensors)
        << " addr=" << addr;
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclSetRawTensorAddr(aclTensor* tensor, void* addr) {
    std::ostringstream log;
    log << "[aclSetRawTensorAddr] tensor=" << static_cast<const void*>(tensor)
        << " addr=" << addr;
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclGetRawTensorAddr(const aclTensor* tensor, void** addr) {
    std::ostringstream log;
    log << "[aclGetRawTensorAddr] tensor=" << static_cast<const void*>(tensor)
        << " addr=" << static_cast<void*>(addr);
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclDumpOpTensors(const char* opType, const char* opName, aclTensor** tensors,
                             size_t inputTensorNum, size_t outputTensorNum, aclrtStream stream) {
    std::ostringstream log;
    log << "[aclDumpOpTensors] opType=" << (opType ? opType : "null")
        << " opName=" << (opName ? opName : "null")
        << " tensors=" << static_cast<const void*>(tensors)
        << " inputTensorNum=" << inputTensorNum
        << " outputTensorNum=" << outputTensorNum
        << " stream=" << stream;
    log_output(log, true);
    return UNIMPLEMENTED;
}


// ~~~ общие компоненты ~~~

struct UnaryExecutor {
    const aclTensor* x;
    const aclTensor* out;
};


// ~~~ aclnn_angle_v2.h ~~~

__attribute__((visibility("default")))
aclnnStatus aclnnAngleV2GetWorkspaceSize(const aclTensor *x,
                                         const aclTensor *out,
                                         uint64_t *workspaceSize,
                                         aclOpExecutor **executor) {
    std::ostringstream log;
    log << "[aclnnAngleV2GetWorkspaceSize] x=" << static_cast<const void*>(x)
        << " out=" << static_cast<const void*>(out)
        << " workspaceSize=" << static_cast<void*>(workspaceSize)
        << " executor=" << static_cast<void*>(executor);

    if (!workspaceSize || !executor) {
        log << "\nError: UNIMPLEMENTED";
        log_output(log, true);
        return UNIMPLEMENTED;
    }

    UnaryExecutor* exec = new UnaryExecutor{x, out};
    *workspaceSize = 123;
    *executor = reinterpret_cast<aclOpExecutor*>(exec);

    log << "\n    executor=" << static_cast<const void*>(exec);
    log_output(log, true);

    return OK;
}

__attribute__((visibility("default")))
aclnnStatus aclnnAngleV2(void *workspace,
                         uint64_t workspaceSize,
                         aclOpExecutor *executor,
                         aclrtStream stream) {
    std::ostringstream log;
    log << "[aclnnAngleV2] workspace=" << workspace
        << " workspaceSize=" << workspaceSize
        << " executor=" << static_cast<const void*>(executor)
        << " stream=" << stream;
    log_output(log, true);
    return OK;
}


#ifdef __cplusplus
}
#endif
