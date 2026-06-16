// ===================================================================
//  not_opapi.cpp – собственная реализация API libopapi (aclnn)
//  в составе проекта not_npu.
//
//  Часть операций Ascend CANN (например, torch.angle) использует
//  не стандартный aclopCompileAndExecute(V2), а отдельное API,
//  экспортируемое библиотекой libopapi.so.  Заголовки этого API
//  находятся в пакете: cann-opbase.run/ops_base/aclnn/acl_meta.h
// ===================================================================

#include "common.h"      // log_output, ...
#include "not_acl.cpp"   // aclCreateDataBuffer, aclCreateTensorDesc
#include "not_opapi_base.h"
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif


#define LOG_MEMORY 0

#define MAKE_OP(...)


// ~~~ конструкторы тензоров ~~~

MAKE_OP(aclnnInplaceNormal(out const aclTensor* selfRef, float mean, float std, int64_t seed, int64_t offset,
                           uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto gen = at::detail::getDefaultCPUGenerator();
    gen.set_current_seed(static_cast<uint64_t>(seed) + static_cast<uint64_t>(offset));
    selfRef.normal_(mean, std, gen);
})
MAKE_OP(aclnnInplaceNormalTensor(out const aclTensor* selfRef, float mean, float std,
                                 const aclTensor* seedTensor, const aclTensor* offsetTensor, int64_t offset,
                                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    int64_t seed_val = seedTensor.item<int64_t>();
    int64_t offset_val = offsetTensor.item<int64_t>();
    uint64_t total_seed = static_cast<uint64_t>(seed_val) + static_cast<uint64_t>(offset_val)
                          + static_cast<uint64_t>(offset);
    auto gen = at::detail::getDefaultCPUGenerator();
    gen.set_current_seed(total_seed);
    selfRef.normal_(mean, std, gen);
})

MAKE_OP(aclnnInplaceRandom(out const aclTensor* selfRef, int64_t from, int64_t to, int64_t seed, int64_t offset,
                           uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto gen = at::detail::getDefaultCPUGenerator();
    gen.set_current_seed(static_cast<uint64_t>(seed) + static_cast<uint64_t>(offset));
    selfRef.random_(from, to, gen);
})
MAKE_OP(aclnnInplaceRandomTensor(out const aclTensor* selfRef, int64_t from, int64_t to,
                                 const aclTensor* seedTensor, const aclTensor* offsetTensor, int64_t offset,
                                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    int64_t seed_val = seedTensor.item<int64_t>();
    int64_t offset_val = offsetTensor.item<int64_t>();
    uint64_t total_seed = static_cast<uint64_t>(seed_val) + static_cast<uint64_t>(offset_val)
                          + static_cast<uint64_t>(offset);
    auto gen = at::detail::getDefaultCPUGenerator();
    gen.set_current_seed(total_seed);
    selfRef.random_(from, to, gen);
})

MAKE_OP(aclnnInplaceUniform(out const aclTensor* selfRef, double from, double to, uint64_t seed, uint64_t offset,
                           uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto gen = at::detail::getDefaultCPUGenerator();
    gen.set_current_seed(seed + offset);
    selfRef.uniform_(from, to, gen);
})
MAKE_OP(aclnnInplaceUniformTensor(out const aclTensor* selfRef, double from, double to,
                                 const aclTensor* seedTensor, const aclTensor* offsetTensor,
                                 uint64_t offset, uint64_t* workspaceSize, aclOpExecutor** executor) {
    uint64_t seed_val = static_cast<uint64_t>(seedTensor.item<int64_t>()) + static_cast<uint64_t>(offsetTensor.item<int64_t>()) + offset;
    auto gen = at::detail::getDefaultCPUGenerator();
    gen.set_current_seed(seed_val);
    selfRef.uniform_(from, to, gen);
})


MAKE_OP(aclnnInplaceZero(out aclTensor* selfRef,
                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.zero_();
})

MAKE_OP(aclnnInplaceOne(out const aclTensor* selfRef,
                        uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.fill_(1);
})

MAKE_OP(aclnnInplaceFillDiagonal(out aclTensor* selfRef, const aclScalar* fillValue, bool wrap,
                                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.fill_diagonal_(fillValue.item(), wrap);
})
MAKE_OP(aclnnInplaceFillScalar(out aclTensor* selfRef, const aclScalar* value,
                                uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.fill_(value.item());
})
MAKE_OP(aclnnInplaceFillTensor(out aclTensor* selfRef, const aclTensor* value,
                                uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.copy_(value);
})

MAKE_OP(aclnnRandperm(int64_t n, int64_t seed, int64_t offset, sync aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto gen = at::detail::getDefaultCPUGenerator();
    gen.set_current_seed(static_cast<uint64_t>(seed) + static_cast<uint64_t>(offset));
    at::randperm_out(out, n, gen);
})

MAKE_OP(aclnnEye(int64_t n, int64_t m, out aclTensor* out,
                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::eye_out(out, n, m);
})

MAKE_OP(aclnnOneHot(const aclTensor* self, int numClasses, const aclTensor* onValue, const aclTensor* offValue, int64_t axis,
                    sync aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    if (numClasses <= 0)
        throw AclnnException(INVALID_PARAM, "numClasses must be positive");
    if (self.numel() == 0)
        return OK;

    auto one_hot = at::one_hot(self.to(at::kLong), numClasses);
    int64_t ndim = self.dim();
    int64_t ax = axis < 0 ? axis + ndim + 1 : axis;
    if (ax != ndim) {
        std::vector<int64_t> perm;
        for (int64_t i = 0; i < ax; ++i)
            perm.push_back(i);
        perm.push_back(ndim);
        for (int64_t i = ax; i < ndim; ++i)
            perm.push_back(i);
        one_hot = one_hot.permute(perm).contiguous();
    }
    out.copy_(one_hot.to(out.options()));
})

MAKE_OP(aclnnSort(const aclTensor* self, bool stable, int64_t dim, bool descending,
                  out aclTensor* valuesOut, out aclTensor* indicesOut,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::sort_out(valuesOut, indicesOut, self, dim, descending);
})


// ~~~ комбинаторы ~~~

MAKE_OP(aclnnCat(const aclTensorList* tensors, int64_t dim, out aclTensor* out,
                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::cat_out(out, tensors, dim);
})

MAKE_OP(aclnnStack(const aclTensorList* tensors, int64_t dim, out aclTensor* out,
                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::stack_out(out, tensors, dim);
})

MAKE_OP(aclnnInplaceCopy(out aclTensor* selfRef, const aclTensor* src,
                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.copy_(src);
})


// ~~~ унарные операции ~~~

MAKE_OP(aclnnAbs(const aclTensor* self, out aclTensor* out,
                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::abs_out(out, self);
})

MAKE_OP(aclnnCeil(const aclTensor* self, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::ceil_out(out, self);
})

MAKE_OP(aclnnIsFinite(const aclTensor* self, out aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    out.copy_(at::isfinite(self));
})

MAKE_OP(aclnnAngleV2(const aclTensor *x, out const aclTensor *out,
                     uint64_t *workspaceSize, aclOpExecutor **executor) {
    at::angle_out(out, x);
})


// ~~~ бинарные операции ~~~

MAKE_OP(aclnnBitwiseAndScalar(const aclTensor* self, const aclScalar* other, out aclTensor* out,
                              uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::bitwise_and_out(out, self, other); // other -> const at::Tensor& (авто-распаковка)
})
MAKE_OP(aclnnBitwiseAndTensor(const aclTensor* self, const aclTensor* other, out aclTensor* out,
                              uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::bitwise_and_out(out, self, other);
})
MAKE_OP(aclnnBitwiseAndTensorOut(const aclTensor* self, const aclTensor* other, out aclTensor* out,
                                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::bitwise_and_out(out, self, other);
})

MAKE_OP(aclnnDiv(const aclTensor* self, const aclTensor* other, out aclTensor* out,
                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::div_out(out, self, other);
})


// ~~~ компараторы ~~~

MAKE_OP(aclnnNeScalar(const aclTensor* self, const aclScalar* other, out aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::ne_out(out, self, other);
})
MAKE_OP(aclnnNeTensor(const aclTensor* self, const aclTensor* other, out aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::ne_out(out, self, other);
})

MAKE_OP(aclnnLtScalar(const aclTensor* self, const aclScalar* other, out aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::less_out(out, self, other);
})
MAKE_OP(aclnnLtTensor(const aclTensor* self, const aclTensor* other, out aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::less_out(out, self, other);
})

MAKE_OP(aclnnGtScalar(const aclTensor* self, const aclScalar* other, out aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::greater_out(out, self, other);
})
MAKE_OP(aclnnGtTensor(const aclTensor* self, const aclTensor* other, out aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::greater_out(out, self, other);
})


// ~~~ редукция ~~~

MAKE_OP(aclnnMin(const aclTensor* self, out aclTensor* out,
                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::min_out(out, self);
})
MAKE_OP(aclnnMinDim(const aclTensor* self, int64_t dim, bool keepdim,
                    out aclTensor* out, out aclTensor* indices,
                    uint64_t* workspaceSize, aclOpExecutor** executor) {
    // indices — выходной тензор индексов (int64)
    at::min_out(out, indices, self, dim, keepdim);
})
MAKE_OP(aclnnMinN(const aclTensorList* tensors, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::Tensor stacked = at::stack(tensors, 0);
    out.copy_(std::get<0>(at::min(stacked, 0)));
})
MAKE_OP(aclnnMinimum(const aclTensor* self, const aclTensor* other, out aclTensor* out,
                     uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::minimum_out(out, self, other);
})

MAKE_OP(aclnnMax(const aclTensor* self, sync aclTensor* out,
                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::max_out(out, self);
})
MAKE_OP(aclnnMaxDim(const aclTensor* self, int64_t dim, bool keepdim,
                    sync aclTensor* out, out aclTensor* indices,
                    uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::max_out(out, indices, self, dim, keepdim);
})
MAKE_OP(aclnnMaxN(const aclTensorList* tensors, sync aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::Tensor stacked = at::stack(tensors, 0);
    out.copy_(std::get<0>(at::max(stacked, 0)));
})
MAKE_OP(aclnnMaximum(const aclTensor* self, const aclTensor* other, out aclTensor* out,
                     uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::maximum_out(out, self, other);
})
MAKE_OP(aclnnMaxV2(const aclTensor* self, const aclIntArray* dims, bool keepDims, bool noopWithEmptyDims,
                   sync aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    if (noopWithEmptyDims && dims.empty())
        out.copy_(self);
    else
        at::amax_out(out, self, dims, keepDims);
})


MAKE_OP(aclnnAll(const aclTensor* self, const aclIntArray* dim, bool keepdim, sync aclTensor* out,
                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::all_out(out, self, dim, keepdim);
})

MAKE_OP(aclnnAny(const aclTensor* self, const aclIntArray* dim, bool keepdim, sync aclTensor* out,
                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::any_out(out, self, dim, keepdim);
})

MAKE_OP(aclnnLogSumExp(const aclTensor* self, const aclIntArray* dim, bool keepDim, sync aclTensor* out,
                       uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::logsumexp_out(out, self, dim, keepDim);
})


MAKE_OP(aclnnAmax(const aclTensor* self, const aclIntArray* dim, bool keepDim, sync aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::amax_out(out, self, dim, keepDim);
})

MAKE_OP(aclnnAmin(const aclTensor* self, const aclIntArray* dim, bool keepDim, sync aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::amin_out(out, self, dim, keepDim);
})

MAKE_OP(aclnnAminmax(const aclTensor* self, const aclIntArray* dim, bool keepDim,
                     sync aclTensor* minOut, sync aclTensor* maxOut,
                     uint64_t* workspaceSize, aclOpExecutor** executor) {
    // Не нашёл способ использовать множественные оси в aten для Aminmax
    at::amin_out(minOut, self, dim, keepDim);
    at::amax_out(maxOut, self, dim, keepDim);
})

MAKE_OP(aclnnAminmaxAll(const aclTensor* self, sync aclTensor* minOut, sync aclTensor* maxOut,
                        uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto result = at::aminmax(self);
    minOut.copy_(std::get<0>(result));
    maxOut.copy_(std::get<1>(result));
})

MAKE_OP(aclnnAminmaxDim(const aclTensor* self, const int64_t dim, bool keepDim,
                        sync aclTensor* minOut, sync aclTensor* maxOut,
                        uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::aminmax_out(minOut, maxOut, self, dim, keepDim);
})


MAKE_OP(aclnnMean(const aclTensor* self, const aclIntArray* dim, bool keepDim, aclDataType dtype,
                  sync aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    if (dtype != self.scalar_type()) {
        if (!dtype)
            return INVALID_PARAM;
        out.copy_(at::mean(self.to(dtype), dim, keepDim));
    } else
        at::mean_out(out, self, dim, keepDim);
})
MAKE_OP(aclnnMeanV2(const aclTensor* self, const aclIntArray* dim, bool keepDim, bool noopWithEmptyAxes,
                    sync aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    if (noopWithEmptyAxes && dim.empty())
        out.copy_(self);
    else
        at::mean_out(out, self, dim, keepDim);
})

MAKE_OP(aclnnReduceLogSum(const aclTensor* data, const aclIntArray* axes, bool keepDims,
                           bool noopWithEmptyAxes, sync aclTensor* reduce,
                           uint64_t* workspaceSize, aclOpExecutor** executor) {
    if (noopWithEmptyAxes && axes.empty())
        reduce.copy_(data);
    else
        at::logsumexp_out(reduce, data, axes, keepDims);
})

MAKE_OP(aclnnReduceNansum(const aclTensor* self, const aclIntArray* dim, bool keepDim,
                          aclDataType dtype, sync aclTensor* out,
                          uint64_t* workspaceSize, aclOpExecutor** executor) {
    if (dtype)
        at::nansum_out(out, self, dim, keepDim, dtype);
    else
        at::nansum_out(out, self, dim, keepDim);
})

MAKE_OP(aclnnReduceSum(const aclTensor* self, const aclIntArray* dims, bool keepDims,
                       aclDataType dtype, sync aclTensor* out,
                       uint64_t* workspaceSize, aclOpExecutor** executor) {
    if (dtype)
        at::sum_out(out, self, dims, keepDims, dtype);
    else
        at::sum_out(out, self, dims, keepDims);
})

MAKE_OP(aclnnVar(const aclTensor* self, const aclIntArray* dim, bool unbiased, bool keepdim,
                 sync aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::var_out(out, self, dim, unbiased, keepdim);
})
MAKE_OP(aclnnVarCorrection(const aclTensor* self, const aclIntArray* dim, int64_t correction, bool keepdim,
                           sync aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::var_out(out, self, dim, correction, keepdim);
})
MAKE_OP(aclnnVarMean(const aclTensor* self, const aclIntArray* dim, int64_t correction, bool keepdim,
                     sync aclTensor* varOut, sync aclTensor* meanOut,
                     uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::var_mean_out(varOut, meanOut, self, dim, correction, keepdim);
})

MAKE_OP(aclnnStd(const aclTensor* self, const aclIntArray* dim, const int64_t correction,
                 bool keepdim, sync aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::std_out(out, self, dim, correction, keepdim);
})
MAKE_OP(aclnnStdMeanCorrection(const aclTensor* self, const aclIntArray* dim,
                               int64_t correction, bool keepdim, sync aclTensor* stdOut, sync aclTensor* meanOut,
                               uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::std_mean_out(stdOut, meanOut, self, dim, correction, keepdim);
})


// ~~~ arg-операции ~~~


// ~~~ операции с маской ~~~

MAKE_OP(aclnnMaskedSelect(const aclTensor* self, const aclTensor* mask, sync aclTensor* out,
                          uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::masked_select_out(out, self, mask);
})


#include "not_opapi_gen.cpp"



// ~~~ Основное meta-API ~~~

aclTensor* aclCreateTensor(const int64_t* viewDims, uint64_t viewDimsNum, aclDataType dataType,
                           const int64_t* stride, int64_t offset, aclFormat format,
                           const int64_t* storageDims, uint64_t storageDimsNum,
                           void* tensorData) {
    std::ostringstream log;
    log << "[aclCreateTensor] dataType=" << aclDataTypeToString(dataType)
        << " format=" << aclFormatToString(format);

    if (viewDimsNum) {
        log << "\n    viewDims: ";
        for (int i = 0; i < viewDimsNum; i++)
            log << (i ? ", " : "") << viewDims[i];

        log << "\n    stride: ";
        for (int i = 0; i < viewDimsNum; i++)
            log << (i ? ", " : "") << stride[i];
    } else
        log << "\n    it's scalar";

    log << "\n    storageDims: ";
    for (int i = 0; i < storageDimsNum; i++)
        log << (i ? ", " : "") << storageDims[i];
    if (!storageDimsNum) log << '-';

    size_t totalElements = 1;
    size_t maxOffset = offset;
    bool negOffset = offset < 0;
    for (uint64_t i = 0; i < viewDimsNum; ++i) {
        totalElements *= viewDims[i];
        size_t localOffset = (viewDims[i] - 1) * stride[i];
        if (localOffset > 0)
            maxOffset += localOffset;
        negOffset |= localOffset < 0;
    }

    if (totalElements > 0) {
        if (storageDimsNum != 1 || maxOffset + 1 > storageDims[0]) {
            log << "\nError: storage size too small: maxOffset=" << maxOffset
                << " >= storageDims[0]=" << (storageDimsNum ? storageDims[0] : '?');
            log_output(log, true);
            return nullptr;
        }
        if (negOffset) {
            log << "\nError: neg offset?!";
            log_output(log, true);
            return nullptr;
        }
    }

    size_t elemSize = aclDataTypeBytes(dataType);
    size_t totalSize = storageDims[0] * elemSize;

    aclTensorDesc* desc = aclCreateTensorDesc(dataType, static_cast<int>(viewDimsNum), viewDims, format);
    aclDataBuffer* buffer = aclCreateDataBuffer(tensorData, totalSize);

    aclTensor* tensor = new aclTensor();
    tensor->desc = desc;
    tensor->buffer = buffer;
    if (stride) {
        tensor->strides.assign(stride, stride + viewDimsNum);
    } else {
        // Плотный stride
        tensor->strides.resize(viewDimsNum);
        int64_t step = 1;
        for (int i = viewDimsNum - 1; i >= 0; --i) {
            tensor->strides[i] = step;
            step *= viewDims[i];
        }
    }
    tensor->offset = offset;

 // log << "\n" << tensorDataToString(tensor);
 // log_output(log, true);

    #if LOG_MEMORY
        std::ostringstream log2;
        log2 << "[NEW TENSOR] addr=" << tensor << " dataType=" << aclDataTypeToString(tensor->desc->dtype);
        log_output(log2, true);
    #endif  // LOG_MEMORY

    return tensor;
}
aclnnStatus aclDestroyTensor(const aclTensor* tensor) {
    #if LOG_MEMORY
        std::ostringstream log;
        log << "[aclDestroyTensor] addr=" << tensor << " dataType=" << aclDataTypeToString(tensor->desc->dtype);
        log_output(log, true);
    #endif  // LOG_MEMORY

    if (tensor) {
        if (tensor->desc)
            aclDestroyTensorDesc(tensor->desc);
        if (tensor->buffer)
            aclDestroyDataBuffer(tensor->buffer);
        delete tensor;
    }
    return OK;
}


aclScalar* aclCreateScalar(void* value, aclDataType dataType) {
    std::ostringstream log;
    log << "[aclCreateScalar] value=" << value
        << " dataType=" << aclDataTypeToString(dataType);

    if (!value) {
        log << "\nError: null scalar value";
        log_output(log, true);
        return nullptr;
    }
    at::ScalarType type;
    if (!try_toAtenType(dataType, type)) {
        log << "\nError: unsupported data type '" << aclDataTypeToString(dataType) << '\'';
        log_output(log, true);
        return nullptr;
    }
 // log_output(log, true);

    auto opts = at::TensorOptions().dtype(type).device(at::kCPU);
    return new aclScalar {
        .tensor = at::from_blob(value, {}, opts).clone(),
        .dtype = dataType,
    };
}
aclnnStatus aclDestroyScalar(const aclScalar* scalar) {
 // std::ostringstream log;
 // log << "[aclDestroyScalar] scalar=" << static_cast<const void*>(scalar);
 // log_output(log, true);

    delete scalar;
    return OK;
}


aclTensorList* aclCreateTensorList(const aclTensor* const* value, uint64_t size) {
    #if LOG_MEMORY
        std::ostringstream log;
        log << "[aclCreateTensorList] tensors=[";
        for (int i = 0; i < size; i++)
            log << (i ? ", " : "") << value[i];
        log << ']';
        log_output(log, true);
    #endif  // LOG_MEMORY

    if (!value && size > 0)
        return nullptr;
    return new aclTensorList(value, size);
}
aclnnStatus aclDestroyTensorList(const aclTensorList* array) {
    #if LOG_MEMORY
        std::ostringstream log;
        log << "[aclDestroyTensorList] array=" << static_cast<const void*>(array);
        log_output(log, true);
    #endif  // LOG_MEMORY

    delete array;
    return OK;
}
aclnnStatus aclGetTensorListSize(const aclTensorList* array, uint64_t* size) {
    std::ostringstream log;
    log << "[aclGetTensorListSize] array=" << static_cast<const void*>(array)
        << " size=" << static_cast<void*>(size);

    if (!array || !size) {
        log << "\nError: invalid param 'array' or 'size'";
        log_output(log, true);
        return INVALID_PARAM;
    }
    log_output(log);

    *size = array->data.size();
    return OK;
}


aclIntArray* aclCreateIntArray(const int64_t* value, uint64_t size) {
    std::ostringstream log;
    log << "[aclCreateIntArray] value=" << static_cast<const void*>(value)
        << " size=" << size;

    if (size && !value) {
        log << "\nError: invalid param 'value'";
        log_output(log, true);
        return nullptr;
    }
    log_output(log);

    auto* arr = new aclIntArray;
    if (size)
        arr->data.assign(value, value + size);
    return arr;
}
aclnnStatus aclDestroyIntArray(const aclIntArray* array) {
    std::ostringstream log;
    log << "[aclDestroyIntArray] array=" << static_cast<const void*>(array);

    if (!array) {
        log << "\nError: invalid param 'array'";
        log_output(log, true);
        return INVALID_PARAM;
    }
    log_output(log);

    delete array;
    return OK;
}
aclnnStatus aclGetIntArraySize(const aclIntArray* array, uint64_t* size) {
    std::ostringstream log;
    log << "[aclGetIntArraySize] array=" << static_cast<const void*>(array)
        << " size=" << static_cast<void*>(size);

    if (!array || !size) {
        log << "\nError: invalid param 'array' or 'size'";
        log_output(log, true);
        return INVALID_PARAM;
    }
    log_output(log);

    *size = array->data.size();
    return OK;
}


aclFloatArray* aclCreateFloatArray(const float* value, uint64_t size) {
    std::ostringstream log;
    log << "[aclCreateFloatArray] value=" << static_cast<const void*>(value)
        << " size=" << size;

    if (size && !value) {
        log << "\nError: invalid param 'value'";
        log_output(log, true);
        return nullptr;
    }
    log_output(log);

    auto* arr = new aclFloatArray;
    if (size)
        arr->data.assign(value, value + size);
    return arr;
}
aclnnStatus aclDestroyFloatArray(const aclFloatArray* array) {
    std::ostringstream log;
    log << "[aclDestroyFloatArray] array=" << static_cast<const void*>(array);

    if (!array) {
        log << "\nError: invalid param 'array'";
        log_output(log, true);
        return INVALID_PARAM;
    }
    log_output(log);

    delete array;
    return OK;
}
aclnnStatus aclGetFloatArraySize(const aclFloatArray* array, uint64_t* size) {
    std::ostringstream log;
    log << "[aclGetFloatArraySize] array=" << static_cast<const void*>(array)
        << " size=" << static_cast<void*>(size);

    if (!array || !size) {
        log << "\nError: invalid param 'array' or 'size'";
        log_output(log, true);
        return INVALID_PARAM;
    }
    log_output(log);

    *size = array->data.size();
    return OK;
}


aclBoolArray* aclCreateBoolArray(const bool* value, uint64_t size) {
    std::ostringstream log;
    log << "[aclCreateBoolArray] value=" << static_cast<const void*>(value)
        << " size=" << size;

    if (size && !value) {
        log << "\nError: invalid param 'value'";
        log_output(log, true);
        return nullptr;
    }
    log_output(log);

    auto* arr = new aclBoolArray;
    if (size)
        arr->data.assign(value, value + size);
    return arr;
}
aclnnStatus aclDestroyBoolArray(const aclBoolArray* array) {
    std::ostringstream log;
    log << "[aclDestroyBoolArray] array=" << static_cast<const void*>(array);

    if (!array) {
        log << "\nError: invalid param 'array'";
        log_output(log, true);
        return INVALID_PARAM;
    }
    log_output(log);

    delete array;
    return OK;
}
aclnnStatus aclGetBoolArraySize(const aclBoolArray* array, uint64_t* size) {
    std::ostringstream log;
    log << "[aclGetBoolArraySize] array=" << static_cast<const void*>(array)
        << " size=" << static_cast<void*>(size);

    if (!array || !size) {
        log << "\nError: invalid param 'array' or 'size'";
        log_output(log, true);
        return INVALID_PARAM;
    }
    log_output(log);

    *size = array->data.size();
    return OK;
}


// TODO: Не совсем понятно, зачем этот aclScalarList,
// если у нас таких операций-то нет, что используют его...

aclScalarList* aclCreateScalarList(const aclScalar* const* value, uint64_t size) {
    std::ostringstream log;
    log << "[aclCreateScalarList] value=" << static_cast<const void*>(value)
        << " size=" << size;

    if (size && !value) {
        log << "\nError: invalid param 'value'";
        log_output(log, true);
        return nullptr;
    }
    log_output(log);

    auto* list = new aclScalarList;
    if (size)
        list->scalars.assign(value, value + size);
    return list;
}
aclnnStatus aclDestroyScalarList(const aclScalarList* array) {
    std::ostringstream log;
    log << "[aclDestroyScalarList] array=" << static_cast<const void*>(array);

    if (!array) {
        log << "\nError: invalid param 'array'";
        log_output(log, true);
        return INVALID_PARAM;
    }
    log_output(log);

    delete array;
    return OK;
}
aclnnStatus aclGetScalarListSize(const aclScalarList* scalarList, uint64_t* size) {
    std::ostringstream log;
    log << "[aclGetScalarListSize] scalarList=" << static_cast<const void*>(scalarList)
        << " size=" << static_cast<void*>(size);

    if (!scalarList || !size) {
        log << "\nError: invalid param 'scalarList' or 'size'";
        log_output(log, true);
        return INVALID_PARAM;
    }
    log_output(log);

    *size = scalarList->scalars.size();
    return OK;
}


aclnnStatus aclGetViewShape(const aclTensor* tensor, int64_t** viewDims, uint64_t* viewDimsNum) {
    std::ostringstream log;
    log << "[aclGetViewShape] tensor=" << static_cast<const void*>(tensor);

    if (!tensor || !tensor->desc || !viewDims || !viewDimsNum) {
        log << " viewDims=" << viewDims << " viewDimsNum=" << viewDimsNum
            << "\nError: invalid argument";
        log_output(log, true);
        return UNIMPLEMENTED;
    }

    const auto& dims = tensor->desc->dims;
 // *viewDims = const_cast<int64_t*>(dims.data());   приведёт к двойному
 // free - после выхода из aclGetViewShape и после освобождения самого aclTensor.
 // ЗНАЧИТ! Необходимо использовать копирование через malloc
    if (dims.empty()) {
        *viewDims = nullptr;
    } else {
        *viewDims = static_cast<int64_t*>(malloc(dims.size() * sizeof(int64_t)));
        if (!*viewDims) {
            log << "\nError: malloc failed";
            log_output(log, true);
            return MEMORY_FAULT;
        }
        std::copy(dims.begin(), dims.end(), *viewDims);
    }
    *viewDimsNum = dims.size();

    log << "\n    dims: ";
    for (size_t i = 0; i < dims.size(); ++i)
        log << (i ? ", " : "") << dims[i];
    log_output(log);

    return OK;
}

aclnnStatus aclGetDataType(const aclTensor* tensor, aclDataType* dataType) {
    std::ostringstream log;
    log << "[aclGetDataType] tensor=" << static_cast<const void*>(tensor)
        << " dataType=" << static_cast<void*>(dataType)
        << "\nError: UNIMPLEMENTED BASE OP";
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclGetStorageShape(const aclTensor* tensor, int64_t** storageDims, uint64_t* storageDimsNum) {
    std::ostringstream log;
    log << "[aclGetStorageShape] tensor=" << static_cast<const void*>(tensor)
        << " storageDims=" << static_cast<void*>(storageDims)
        << " storageDimsNum=" << static_cast<void*>(storageDimsNum)
        << "\nError: UNIMPLEMENTED BASE OP";
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclGetViewStrides(const aclTensor* tensor, int64_t** stridesValue, uint64_t* stridesNum) {
    std::ostringstream log;
    log << "[aclGetViewStrides] tensor=" << static_cast<const void*>(tensor)
        << " stridesValue=" << static_cast<void*>(stridesValue)
        << " stridesNum=" << static_cast<void*>(stridesNum)
        << "\nError: UNIMPLEMENTED BASE OP";
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclGetViewOffset(const aclTensor* tensor, int64_t* offset) {
    std::ostringstream log;
    log << "[aclGetViewOffset] tensor=" << static_cast<const void*>(tensor)
        << " offset=" << static_cast<void*>(offset)
        << "\nError: UNIMPLEMENTED BASE OP";
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclGetFormat(const aclTensor* tensor, aclFormat* format) {
    std::ostringstream log;
    log << "[aclGetFormat] tensor=" << static_cast<const void*>(tensor)
        << " format=" << static_cast<void*>(format)
        << "\nError: UNIMPLEMENTED BASE OP";
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
        << " tensorDataAddr=" << tensorDataAddr
        << "\nError: UNIMPLEMENTED BASE OP";
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclSetAclOpExecutorRepeatable(aclOpExecutor* executor) {
    std::ostringstream log;
    log << "[aclSetAclOpExecutorRepeatable] executor=" << static_cast<const void*>(executor)
        << "\nError: UNIMPLEMENTED BaseOP";
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclDestroyAclOpExecutor(aclOpExecutor* executor) {
    std::ostringstream log;
    log << "[aclDestroyAclOpExecutor] executor=" << static_cast<const void*>(executor)
        << "\nError: UNIMPLEMENTED BaseOP";
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
        << " addr=" << addr \
        << "\nError: UNIMPLEMENTED BaseOP"; \
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
        << " addr=" << addr \
        << "\nError: UNIMPLEMENTED BaseOP";
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
        << " addr=" << addr \
        << "\nError: UNIMPLEMENTED BaseOP";
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
        << " addr=" << addr \
        << "\nError: UNIMPLEMENTED BaseOP";
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
        << " addr=" << addr \
        << "\nError: UNIMPLEMENTED BaseOP";
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclSetRawTensorAddr(aclTensor* tensor, void* addr) {
    std::ostringstream log;
    log << "[aclSetRawTensorAddr] tensor=" << static_cast<const void*>(tensor)
        << " addr=" << addr \
        << "\nError: UNIMPLEMENTED BaseOP";
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclGetRawTensorAddr(const aclTensor* tensor, void** addr) {
    std::ostringstream log;
    log << "[aclGetRawTensorAddr] tensor=" << static_cast<const void*>(tensor)
        << " addr=" << static_cast<void*>(addr) \
        << "\nError: UNIMPLEMENTED BaseOP";
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
        << " stream=" << stream \
        << "\nError: UNIMPLEMENTED BaseOP";
    log_output(log, true);
    return UNIMPLEMENTED;
}



// ~~~ UNIMPLEMENTED OPS ~~~

/*
function where_op() {
    grep "$1" ~/tmp/Ascend-cann-950/run_package/ -rn --include="*.h"
    echo "~~~~~~~~~~~~~~~~"
    grep "$1" ~/tmp/pytorch/third_party/op-plugin/ -rn
}
where_op aclnnGelu
*/

#define DEFINE_UNIMPLEMENTED_ACLNN(name, ...)                          \
    __attribute__((visibility("default"), weak))                       \
    aclnnStatus name(__VA_ARGS__) {                                    \
        log_output("[" #name "]\nError: UNIMPLEMENTED " #name, true);  \
        return UNIMPLEMENTED;                                          \
    };
// ТАК ВОТ ЗАЧЕМ РАЗРАБОТЧИКИ ОРИГИНАЛЬНОГО ACLNN ИСПОЛЬЗОВАЛИ __attribute__!!!
// Ставим weak в заглушку и она перестаёт конфликтовать с нормальными реализациями!!!

DEFINE_UNIMPLEMENTED_ACLNN(aclStftGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* windowOptional, aclTensor* out, int64_t nFft, int64_t hopLength,
                           int64_t winLength, bool normalized, bool onesided, bool returnComplex, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclStft,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAcosGetWorkspaceSize,
                           const aclTensor* input, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAcos,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAcoshGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAcosh,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAdaLayerNormGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* scale,
                           const aclTensor* shift, const aclTensor* weightOptional,
                           const aclTensor* biasOptional, double epsilon, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAdaLayerNorm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAdaLayerNormQuantGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* scale, const aclTensor* shift, const aclTensor* weightOptional,
                           const aclTensor* biasOptional, const aclTensor* smoothScalesOptional, double epsilon, const char* quantMode,
                           aclTensor* out, aclTensor* quantScale, aclTensor* quantOffsetOptional, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAdaLayerNormQuant,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAdaLayerNormV2GetWorkspaceSize,
                           const aclTensor* x, const aclTensor* scale,
                           const aclTensor* shift, const aclTensor* weightOptional,
                           const aclTensor* biasOptional, double epsilon, aclTensor* out,
                           aclTensor* meanOutOptional, aclTensor* rstdOutOptional,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAdaLayerNormV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAdaptiveAvgPool2dGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* outputSize, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAdaptiveAvgPool2d,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAdaptiveAvgPool2dBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAdaptiveAvgPool2dBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAdaptiveAvgPool3dGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* outputSize, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAdaptiveAvgPool3d,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAdaptiveAvgPool3dBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAdaptiveAvgPool3dBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAdaptiveMaxPool2dGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* outputSize, aclTensor* outputOut, aclTensor* indicesOut,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAdaptiveMaxPool2d,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, aclTensor* gradInput,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAdaptiveMaxPool2dBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAdaptiveMaxPool3dGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* outputSize, aclTensor* outputOut, aclTensor* indicesOut,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAdaptiveMaxPool3d,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAdaptiveMaxPool3dBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, aclTensor* gradInput,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAdaptiveMaxPool3dBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* other, const aclScalar* alpha, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAdd,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddLayerNormQuantGetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* beta,
                           const aclTensor* biasOptional, const aclTensor* scales1Optional, const aclTensor* scales2Optional,
                           const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional, const char* quantMode, double epsilon,
                           bool additionalOutput, bool divMode, aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut, aclTensor* outScales1Out,
                           aclTensor* outScales2Out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddLayerNormQuant,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddReluGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* other, aclScalar* alpha,
                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddRelu,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddRmsNormGetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, double epsilon, aclTensor* yOut,
                           aclTensor* rstdOut, aclTensor* xOut, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddRmsNorm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddRmsNormDynamicQuantGetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* smoothScale1Optional,
                           const aclTensor* smoothScale2Optional, double epsilon, aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut,
                           aclTensor* scale1Out, aclTensor* scale2Out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddRmsNormDynamicQuant,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddRmsNormDynamicQuantV2GetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* smoothScale1Optional,
                           const aclTensor* smoothScale2Optional, const aclTensor* betaOptional, double epsilon,
                           const aclBoolArray* outputMask, aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut, aclTensor* scale1Out,
                           aclTensor* scale2Out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddRmsNormDynamicQuantV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddRmsNormQuantGetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* scales1,
                           const aclTensor* scales2Optional, const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional,
                           int64_t axis, double epsilon, bool divMode, aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddRmsNormQuant,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddRmsNormQuantV2GetWorkspaceSize,
                           const aclTensor *x1, const aclTensor *x2, const aclTensor *gamma,
                           const aclTensor *scales1, const aclTensor *scales2Optional,
                           const aclTensor *zeroPoints1Optional, const aclTensor *zeroPoints2Optional,
                           const aclTensor *betaOptional,
                           int64_t axis, double epsilon, bool divMode,
                           aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut, aclTensor* rmsNormOut,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddRmsNormQuantV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddV3GetWorkspaceSize,
                           const aclScalar* self, const aclTensor* other, const aclScalar* alpha, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddV3,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddbmmGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* batch1, const aclTensor* batch2, const aclScalar* beta,
                           const aclScalar* alpha, aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddbmm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddcdivGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2, const aclScalar* value,
                           const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddcdiv,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddcmulGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2, const aclScalar* value, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddcmul,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddmmGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* mat1, const aclTensor* mat2, const aclScalar* beta, const aclScalar* alpha,
                           aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddmm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddmmWeightNzGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* mat1, const aclTensor* mat2, const aclScalar* beta, const aclScalar* alpha,
                           aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddmmWeightNz,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddmvGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* mat, const aclTensor* vec, const aclScalar* alpha, const aclScalar* beta,
                           aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddmv,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddrGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* vec1, const aclTensor* vec2, const aclScalar* betaOptional,
                           const aclScalar* alphaOptional, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddr,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddsGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* other, const aclScalar* alpha, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAdds,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAffineGridGetWorkspaceSize,
                           const aclTensor* theta, const aclIntArray* size,
                           bool alignCorners, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAffineGrid,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAllGatherMatmulGetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2,
                           const aclTensor* bias, const char* group,
                           int64_t gatherIndex, int64_t commTurn, int64_t streamMode,
                           const aclTensor* output, const aclTensor* gatherOut,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAllGatherMatmul,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAllGatherMatmulV2GetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                           const aclTensor* x1Scale, const aclTensor* x2Scale,
                           const aclTensor* quantScale, int64_t blockSize, const char* group,
                           int64_t gatherIndex, int64_t commTurn, int64_t streamMode,
                           int64_t groupSize, const char* commMode, aclTensor* output, aclTensor* gatherOut,
                           aclTensor* amaxOut, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAllGatherMatmulV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAlltoAllAllGatherBatchMatMulGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* weight,
                           const aclTensor* biasOptional,
                           const char* groupEp, const char* groupTp,
                           int64_t epWorldSize, int64_t tpWorldSize,
                           int64_t xShardType, int64_t actType,
                           aclTensor* y1Out, aclTensor* y2OutOptional,
                           aclTensor* y3OutOptional,
                           uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAlltoAllAllGatherBatchMatMul,
                           void* workspace, uint64_t workspaceSize,
                           aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAlltoAllvGroupedMatMulGetWorkspaceSize,
                           const aclTensor* gmmX, const aclTensor* gmmWeight,
                           const aclTensor* sendCountsTensorOptional,
                           const aclTensor* recvCountsTensorOptional,
                           const aclTensor* mmXOptional, const aclTensor* mmWeightOptional,
                           const char* group, int64_t epWorldSize,
                           const aclIntArray* sendCounts, const aclIntArray* recvCounts,
                           bool transGmmWeight, bool transMmWeight, bool permuteOutFlag,
                           aclTensor* gmmY, aclTensor* mmYOptional, aclTensor* permuteOutOptional,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAlltoAllvGroupedMatMul,
                           void* workspace, uint64_t workspaceSize,
                           aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnApplyAdamWGetWorkspaceSize,
                           aclTensor* varRef, aclTensor* mRef, aclTensor* vRef,
                           const aclTensor* beta1Power, const aclTensor* beta2Power, const aclTensor* lr,
                           const aclTensor* weightDecay, const aclTensor* beta1, const aclTensor* beta2,
                           const aclTensor* eps, const aclTensor* grad, const aclTensor* maxGradNormOptional,
                           bool amsgrad, bool maximize,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnApplyAdamW,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnApplyAdamWV2GetWorkspaceSize,
                           aclTensor* varRef, aclTensor* mRef, aclTensor* vRef, aclTensor* maxGradNormOptionalRef, const aclTensor* grad,
                           const aclTensor* step, float lr, float beta1, float beta2, float weightDecay, float eps, bool amsgrad,
                           bool maximize, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnApplyAdamWV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnApplyRotaryPosEmbGetWorkspaceSize,
                           aclTensor* queryRef, aclTensor* keyRef, const aclTensor* cos, const aclTensor* sin, int64_t layout,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnApplyRotaryPosEmb,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnApplyRotaryPosEmbV2GetWorkspaceSize,
                           aclTensor* queryRef, aclTensor* keyRef,
                           const aclTensor* cos, const aclTensor* sin,
                           int64_t layout, char* rotaryMode,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnApplyRotaryPosEmbV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnApplyTopKTopPGetWorkspaceSize,
                           const aclTensor* logits, const aclTensor* p,
                           const aclTensor* k, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnApplyTopKTopP,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnArangeGetWorkspaceSize,
                           const aclScalar* start, const aclScalar* end, const aclScalar* step, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnArange,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnArgMaxGetWorkspaceSize,
                           const aclTensor* self, int64_t dim, bool keepdim,
                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnArgMax,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnArgMinGetWorkspaceSize,
                           const aclTensor* self, int64_t dim, bool keepdim,
                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnArgMin,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnArgsortGetWorkspaceSize,
                           const aclTensor* self, int64_t dim, bool descending, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnArgsort,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAscendAntiQuantGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* scale, const aclTensor* offset, int64_t dstType, bool sqrtMode,
                           const aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAscendAntiQuant,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAscendQuantGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* scale, const aclTensor* offset, bool sqrtMode, const char* roundMode,
                           int32_t dstType, const aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAscendQuant,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAscendQuantV3GetWorkspaceSize,
                           const aclTensor* x, const aclTensor* scale, const aclTensor* offset, bool sqrtMode, const char* roundMode,
                           int32_t dstType, int32_t axis, const aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAscendQuantV3,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAsinGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAsin,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAsinhGetWorkspaceSize,
                           const aclTensor* input, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAsinh,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAtanGetWorkspaceSize,
                           const aclTensor* input, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAtan,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAtan2GetWorkspaceSize,
                           const aclTensor* self, const aclTensor* other, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAtan2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAtanhGetWorkspaceSize,
                           const aclTensor* input, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAtanh,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAttentionToFFNGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* sessionId, const aclTensor* microBatchId,
                           const aclTensor* layerId, const aclTensor* expertIds, const aclTensor* expertRankTable,
                           const aclTensor* scales, const aclTensor* activeMask, const char* group, int64_t worldSize,
                           const aclIntArray *ffnTokenInfoTableShape, const aclIntArray *ffnTokenDataShape,
                           const aclIntArray *attnTokenInfoTableShape, int64_t moeExpertNum, int64_t quantMode, int64_t syncFlag,
                           int64_t ffnStartRankId, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAttentionToFFN,
                           void* workspace, uint64_t workspaceSize,
                           aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAttentionUpdateGetWorkspaceSize,
                           const aclTensorList* lse, const aclTensorList* localOut, int64_t updateType,
                           aclTensor* out, aclTensor* lseOut, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAttentionUpdate,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAvgPool2dGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* strides, const aclIntArray* paddings,
                           const bool ceilMode, const bool countIncludePad, const int64_t divisorOverride, const int8_t cubeMathType,
                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAvgPool2d,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAvgPool2dBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride,
                           const aclIntArray* padding, bool ceilMode, bool countIncludePad, int64_t divisorOverride, int8_t cubeMathType,
                           aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAvgPool2dBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAvgPool3dGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding,
                           bool ceilMode, bool countIncludePad, int64_t divisorOverride, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAvgPool3d,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAvgPool3dBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride,
                           const aclIntArray* padding, bool ceilMode, bool countIncludePad, int64_t divisorOverride, aclTensor* output,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAvgPool3dBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBaddbmmGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* batch1, const aclTensor* batch2, const aclScalar* beta,
                           const aclScalar* alpha, aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBaddbmm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBatchMatMulGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* mat2, aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBatchMatMul,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* weight,
                           const aclTensor* biasOptional,
                           const char* groupEp, const char* groupTp,
                           int64_t epWorldSize, int64_t tpWorldSize,
                           int64_t yShardType, aclTensor* out,
                           uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBatchMatMulReduceScatterAlltoAll,
                           void* workspace, uint64_t workspaceSize,
                           aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBatchMatMulWeightNzGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* mat2, aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBatchMatMulWeightNz,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBatchMatmulQuantGetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* quantParam, const aclTensor* bias, bool transposeX1,
                           bool transposeX2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBatchMatmulQuant,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBatchNormGetWorkspaceSize,
                           const aclTensor* input, const aclTensor* weight, const aclTensor* bias, aclTensor* runningMean,
                           aclTensor* runningVar, bool training, double momentum, double eps, aclTensor* output, aclTensor* saveMean,
                           aclTensor* saveInvstd, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBatchNorm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBatchNormBackwardGetWorkspaceSize,
                           const aclTensor* gradOut, const aclTensor* input, const aclTensor* weight, const aclTensor* runningMean,
                           const aclTensor* runningVar, const aclTensor* saveMean, const aclTensor* saveInvstd, bool training, double eps,
                           const aclBoolArray* outputMask, aclTensor* gradInput, aclTensor* gradWeight, aclTensor* gradBias,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBatchNormBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBatchNormElemtGetWorkspaceSize,
                           const aclTensor* input, const aclTensor* weight, const aclTensor* bias, aclTensor* mean, aclTensor* invstd,
                           double eps, aclTensor* output, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBatchNormElemt,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBatchNormElemtBackwardGetWorkspaceSize,
                           const aclTensor* gradOut, const aclTensor* input, const aclTensor* mean, const aclTensor* invstd,
                           const aclTensor* weight, const aclTensor* sumDy, const aclTensor* sumDyXmu, aclTensor* counter,
                           aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBatchNormElemtBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBatchNormGatherStatsWithCountsGetWorkspaceSize,
                           const aclTensor* input, const aclTensor* mean, const aclTensor* invstd, aclTensor* runningMean,
                           aclTensor* runningVar, double momentum, double eps, const aclTensor* counts, aclTensor* meanAll,
                           aclTensor* invstdAll, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBatchNormGatherStatsWithCounts,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBatchNormReduceGetWorkspaceSize,
                           const aclTensor* x, aclTensor* sum, aclTensor* squareSum, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBatchNormReduce,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBatchNormReduceBackwardGetWorkspaceSize,
                           const aclTensor* gradOut, const aclTensor* input, const aclTensor* mean, const aclTensor* invstd,
                           const aclTensor* weight, const bool inputG, const bool weightG, const bool biasG, aclTensor* sumDy,
                           aclTensor* sumDyXmu, aclTensor* gradWeight, aclTensor* gradBias, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBatchNormReduceBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBatchNormStatsGetWorkspaceSize,
                           const aclTensor* input, double eps, aclTensor* mean, aclTensor* invstd, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBatchNormStats,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBernoulliGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* prob, int64_t seed, int64_t offset, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBernoulli,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBernoulliTensorGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* prob, int64_t seed, int64_t offset, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBernoulliTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBinaryCrossEntropyGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* target,
                           const aclTensor* weight, int64_t reduction,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBinaryCrossEntropy,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBinaryCrossEntropyBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, const aclTensor* weightOptional,
                           int64_t reduction, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBinaryCrossEntropyBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBinaryCrossEntropyWithLogitsGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* target, const aclTensor* weightOptional, const aclTensor* posWeightOptional,
                           int64_t reduction, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBinaryCrossEntropyWithLogits,
                           void* workspace, uint64_t workspaceSize,
                           aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBinaryCrossEntropyWithLogitsBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, const aclTensor* weightOptional,
                           const aclTensor* posWeightOptional, int64_t reduction, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBinaryCrossEntropyWithLogitsBackward,
                           void* workspace, uint64_t workspaceSize,
                           aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, const aclTensor* weightOptional,
                           const aclTensor* posWeightOptional, int64_t reduction, aclTensor* gradTarget, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBinaryCrossEntropyWithLogitsTargetBackward,
                           void* workspace, uint64_t workspaceSize,
                           aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBincountGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* weights, int64_t minlength, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBincount,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBitwiseNotGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBitwiseNot,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBitwiseOrScalarGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBitwiseOrScalar,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBitwiseOrTensorGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBitwiseOrTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBitwiseXorScalarGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBitwiseXorScalar,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBitwiseXorTensorGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBitwiseXorTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnCIoUGetWorkspaceSize,
                           const aclTensor* bBoxes, const aclTensor* gtBoxes, bool trans, bool isCross, const char* mode, aclTensor* overlap,
                           aclTensor* atanSub, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnCIoU,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnCalculateConvolutionWeightSize,
                           const aclIntArray* tensorShape, bool transposed,
                           int64_t groups, aclDataType dataType,
                           uint64_t* weightTensorSize)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnCalculateMatmulWeightSize,
                           const aclIntArray* tensorShape, uint64_t* weightTensorSize)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnCalculateMatmulWeightSizeV2,
                           const aclIntArray* tensorShape, aclDataType dataType,
                           uint64_t* weightTensorSize)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnCastGetWorkspaceSize,
                           const aclTensor* self, const aclDataType dtype, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnCast,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnCdistGetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2,
                           float p, int64_t compute_mode, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnCdist,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnCdistBackwardGetWorkspaceSize,
                           const aclTensor* grad, const aclTensor* x1, const aclTensor* x2, const aclTensor* cdist, float p,
                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnCdistBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnCeluGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* alpha, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnCelu,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnChamferDistanceBackwardGetWorkspaceSize,
                           const aclTensor* xyz1, const aclTensor* xyz2, const aclTensor* idx1, const aclTensor* idx2,
                           const aclTensor* gradDist1, const aclTensor* gradDist2, aclTensor* gradXyz1, aclTensor* gradXyz2,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnChamferDistanceBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnChannelShuffleGetWorkspaceSize,
                           const aclTensor* self, int64_t groups, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnChannelShuffle,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnCircularPad2dGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* padding, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnCircularPad2d,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnCircularPad2dBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* padding, aclTensor* gradInput,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnCircularPad2dBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnCircularPad3dGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* padding, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnCircularPad3d,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnCircularPad3dBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* padding, aclTensor* gradInput,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnCircularPad3dBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnClampGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* clipValueMin, const aclScalar* clipValueMax, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnClamp,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnClampMaxGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* clipValueMax, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnClampMax,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnClampMaxTensorGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* max, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnClampMaxTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnClampMinGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* clipValueMin, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnClampMin,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnClampMinTensorGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* clipValueMin, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnClampMinTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnClampTensorGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* clipValueMin, const aclTensor* clipValueMax, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnClampTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnComplexGetWorkspaceSize,
                           const aclTensor* real, const aclTensor* imag, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnComplex,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnConstantPadNdGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* pad,
                           const aclScalar* value, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnConstantPadNd,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnConvDepthwise2dGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* weight,
                           const aclIntArray* kernelSize, const aclTensor* bias,
                           const aclIntArray* stride, const aclIntArray* padding,
                           const aclIntArray* dilation, aclTensor* out,
                           int8_t cubeMathType, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnConvDepthwise2d,
                           void* workspace, const uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnConvTbcGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* weight,
                           const aclTensor* bias, const int64_t pad, aclTensor* output,
                           int8_t cubeMathType, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnConvTbc,
                           void* workspace, const uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnConvTbcBackwardGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* input,
                           const aclTensor* weight, const aclTensor* bias,
                           int64_t pad, int8_t cubeMathType,
                           aclTensor* gradInput, aclTensor* gradWeight,
                           aclTensor* gradBias, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnConvTbcBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnConvertWeightToINT4PackGetWorkspaceSize,
                           const aclTensor *weight, aclTensor *weightInt4Pack,
                           uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnConvertWeightToINT4Pack,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnConvolutionGetWorkspaceSize,
                           const aclTensor* input, const aclTensor* weight,
                           const aclTensor* bias, const aclIntArray* stride,
                           const aclIntArray* padding, const aclIntArray* dilation,
                           bool transposed, const aclIntArray* outputPadding,
                           const int64_t groups, aclTensor* output, int8_t cubeMathType,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnConvolution,
                           void* workspace, const uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnConvolutionBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* input, const aclTensor* weight, const aclIntArray* biasSizes,
                           const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation, bool transposed,
                           const aclIntArray* outputPadding, int groups, const aclBoolArray* outputMask, int8_t cubeMathType,
                           aclTensor* gradInput, aclTensor* gradWeight, aclTensor* gradBias, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnConvolutionBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnCosGetWorkspaceSize,
                           const aclTensor* input, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnCos,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnCoshGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnCosh,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnCtcLossGetWorkspaceSize,
                           const aclTensor* logProbs, const aclTensor* targets, const aclIntArray* inputLengths,
                           const aclIntArray* targetlengths, int64_t blank, bool zeroInfinity, aclTensor* negLogLikelihoodOut,
                           aclTensor* logAlphaOut, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnCtcLoss,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnCtcLossBackwardGetWorkspaceSize,
                           const aclTensor* gradOut, const aclTensor* logProbs, const aclTensor* targets, const aclIntArray* inputLengths,
                           const aclIntArray* targetLengths, const aclTensor* negLogLikelihood, const aclTensor* logAlpha, int64_t blank,
                           bool zeroInfinity, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnCtcLossBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnCummaxGetWorkspaceSize,
                           const aclTensor* self, int64_t dim, aclTensor* valuesOut,
                           aclTensor* indicesOut, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnCummax,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnCumminGetWorkspaceSize,
                           const aclTensor* self, int64_t dim, aclTensor* valuesOut,
                           aclTensor* indicesOut, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnCummin,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnCumprodGetWorkspaceSize,
                           const aclTensor *input, const aclScalar *dim, const aclDataType dtype,
                           aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnCumprod,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnCumsumGetWorkspaceSize,
                           const aclTensor* self, int64_t dim, aclDataType dtype, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnCumsum,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnCumsumV2GetWorkspaceSize,
                           const aclTensor* self, int64_t dim, bool exclusive, bool reverse,
                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnCumsumV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnDeformableConv2dGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* weight,
                           const aclTensor* offset, const aclTensor* biasOptional,
                           const aclIntArray* kernelSize, const aclIntArray* stride,
                           const aclIntArray* padding, const aclIntArray* dilation,
                           int64_t groups, int64_t deformableGroups, bool modulated,
                           aclTensor* out, aclTensor* deformOutOptional,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnDeformableConv2d,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnDiagGetWorkspaceSize,
                           const aclTensor* self, int64_t diagonal, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnDiag,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnDiagFlatGetWorkspaceSize,
                           const aclTensor* self, int64_t diagonal, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnDiagFlat,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnDigammaGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnDigamma,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnDivModGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* other, int mode, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnDivMod,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnDivModsGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* other, int mode, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnDivMods,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnDivsGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnDivs,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnDotGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* tensor, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnDot,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnDropoutGetWorkspaceSize,
                           const aclTensor* input, double p, bool train, int64_t seed, int64_t offset, aclTensor* out, aclTensor* maskOut,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnDropout,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnDropoutBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* mask, double scale, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnDropoutBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnDropoutDoMaskGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* mask, double prob, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnDropoutDoMask,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnDropoutGenMaskGetWorkspaceSize,
                           const aclIntArray* shape, double prob, int64_t seed, int64_t offset, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnDropoutGenMask,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnDropoutGenMaskV2GetWorkspaceSize,
                           const aclIntArray* shape, double prob, int64_t seed, int64_t offset, aclDataType probDataType, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnDropoutGenMaskV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnDropoutGenMaskV2TensorGetWorkspaceSize,
                           const aclIntArray* shape, double prob, const aclTensor* seedTensor, const aclTensor* offsetTensor, int64_t offset,
                           aclDataType probDataType, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnDropoutGenMaskV2Tensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnDropoutV3GetWorkspaceSize,
                           const aclTensor* input, const aclTensor* optionalNoiseShape, double p, int64_t seed, int64_t offset, aclTensor* out,
                           aclTensor* maskOut, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnDropoutV3,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnDualLevelQuantMatmulWeightNzGetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Level0Scale, const aclTensor* x2Level0Scale,
                           const aclTensor* x1Level1Scale, const aclTensor* x2Level1Scale, const aclTensor* optionalBias, bool transposeX1,
                           bool transposeX2, int64_t level0GroupSize, int64_t level1GroupSize, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnDualLevelQuantMatmulWeightNz,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnEinsumGetWorkspaceSize,
                           const aclTensorList* tensors, const char* equation, aclTensor* output,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnEinsum,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnEluGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* alpha, const aclScalar* scale, const aclScalar* inputScale, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnElu,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnEluBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclScalar* alpha,
                           const aclScalar* scale, const aclScalar* inputScale,
                           bool isResult, const aclTensor* selfOrResult,
                           aclTensor* gradInput, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnEluBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnEmbeddingGetWorkspaceSize,
                           const aclTensor* weight, const aclTensor* indices,
                           const aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnEmbedding,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnEmbeddingBagGetWorkspaceSize,
                           const aclTensor* weight, const aclTensor* indices, const aclTensor* offsets, bool scaleGradByFreq, int64_t mode,
                           bool sparse, const aclTensor* perSampleWeights, bool includeLastOffset, int64_t paddingIdx, aclTensor* output,
                           aclTensor* offset2bag, aclTensor* bagSize, aclTensor* maxIndices, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnEmbeddingBag,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnEmbeddingDenseBackwardGetWorkspaceSize,
                           const aclTensor* grad, const aclTensor* indices, uint64_t numWeights, uint64_t paddingIdx, bool scaleGradByFreq,
                           const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnEmbeddingDenseBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnEmbeddingRenormGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* indices, double maxNorm,
                           double normType, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnEmbeddingRenorm,
                           void* workspace, uint64_t workspace_size, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnEqScalarGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnEqScalar,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnEqTensorGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnEqTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnEqualGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* other, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnEqual,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnErfGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnErf,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnErfcGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnErfc,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnErfinvGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnErfinv,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnExpGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnExp,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnExp2GetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnExp2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnExpSegsumGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnExpSegsum,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnExpSegsumBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput,const aclTensor* gradSelf,
                           aclTensor* gradInput, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnExpSegsumBackward,
                           void* workspace, uint64_t workspaceSize,
                           aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnExpandGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* size, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnExpand,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnExpm1GetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnExpm1,
                           void* workspace, uint64_t workspace_size, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnFFNToAttentionGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* sessionIds,
                           const aclTensor* microBatchIds, const aclTensor* tokenIds, const aclTensor* expertOffsets,
                           const aclTensor* actualTokenNum, const aclTensor* attnRankTable, const char* group, int64_t worldSize,
                           const aclIntArray *tokenInfoTableShape, const aclIntArray *tokenDataShape,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnFFNToAttention,
                           void* workspace, uint64_t workspaceSize,
                           aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnFakeQuantPerChannelAffineCachemaskGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* scale, const aclTensor* zeroPoint, int64_t axis, int64_t quantMin,
                           int64_t quantMax, aclTensor* out, aclTensor* mask, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnFakeQuantPerChannelAffineCachemask,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnFakeQuantPerTensorAffineCachemaskGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* scale, const aclTensor* zeroPoint, float fakeQuantEnbled, int64_t quantMin,
                           int64_t quantMax, aclTensor* out, aclTensor* mask, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnFakeQuantPerTensorAffineCachemask,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnFastBatchNormBackwardGetWorkspaceSize,
                           const aclTensor* gradOut, const aclTensor* input, const aclTensor* weight, const aclTensor* runningMean,
                           const aclTensor* runningVar, const aclTensor* saveMean, const aclTensor* saveInvstd, bool training, double eps,
                           const aclBoolArray* outputMask, int version, aclTensor* gradInput, aclTensor* gradWeight, aclTensor* gradBias,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnFastBatchNormBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnFastGeluGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnFastGelu,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnFastGeluBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, aclTensor* gradInput, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnFastGeluBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnFastLayerNormGetWorkspaceSize,
                           const aclTensor* input, const aclIntArray* normalizedShape, const aclTensor* weightOptional,
                           const aclTensor* biasOptional, double eps, aclTensor* out, aclTensor* meanOutOptional, aclTensor* rstdOutOptional,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnFastLayerNorm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnFlatQuantGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* kroneckerP1,
                           const aclTensor* kroneckerP2, double clipRatio, aclTensor* out,
                           aclTensor* quantScale, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnFlatQuant,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnFlattenGetWorkspaceSize,
                           const aclTensor* self, int64_t axis, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnFlatten,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnFlipGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* dims, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnFlip,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnFloorGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnFloor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnFloorDivideGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnFloorDivide,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnFloorDividesGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnFloorDivides,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnFmodScalarGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnFmodScalar,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnFmodTensorGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnFmodTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnForeachAddListV2GetWorkspaceSize,
                           const aclTensorList *x1,
                           const aclTensorList *x2,
                           const aclScalar *alpha,
                           aclTensorList *out,
                           uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnForeachAddListV2,
                           void *workspace,
                           uint64_t workspaceSize,
                           aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnForeachAddScalarV2GetWorkspaceSize,
                           const aclTensorList *x,
                           const aclScalar *scalar,
                           aclTensorList *out,
                           uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnForeachAddScalarV2,
                           void *workspace,
                           uint64_t workspaceSize,
                           aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnForeachAddcdivScalarV2GetWorkspaceSize,
                           const aclTensorList *x1,
                           const aclTensorList *x2,
                           const aclTensorList *x3,
                           const aclScalar *scalar,
                           aclTensorList *out,
                           uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnForeachAddcdivScalarV2,
                           void *workspace,
                           uint64_t workspaceSize,
                           aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnForeachAddcmulScalarV2GetWorkspaceSize,
                           const aclTensorList *x1,
                           const aclTensorList *x2,
                           const aclTensorList *x3,
                           const aclScalar *scalar,
                           aclTensorList *out,
                           uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnForeachAddcmulScalarV2,
                           void *workspace,
                           uint64_t workspaceSize,
                           aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnForeachDivScalarV2GetWorkspaceSize,
                           const aclTensorList *x,
                           const aclScalar *scalar,
                           aclTensorList *out,
                           uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnForeachDivScalarV2,
                           void *workspace,
                           uint64_t workspaceSize,
                           aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnForeachMaximumScalarV2GetWorkspaceSize,
                           const aclTensorList *x,
                           const aclScalar *scalar,
                           aclTensorList *out,
                           uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnForeachMaximumScalarV2,
                           void *workspace,
                           uint64_t workspaceSize,
                           aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnForeachMinimumScalarV2GetWorkspaceSize,
                           const aclTensorList *x,
                           const aclScalar *scalar,
                           aclTensorList *out,
                           uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnForeachMinimumScalarV2,
                           void *workspace,
                           uint64_t workspaceSize,
                           aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnForeachMulScalarV2GetWorkspaceSize,
                           const aclTensorList *x,
                           const aclScalar *scalar,
                           aclTensorList *out,
                           uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnForeachMulScalarV2,
                           void *workspace,
                           uint64_t workspaceSize,
                           aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnForeachPowScalarV2GetWorkspaceSize,
                           const aclTensorList *x,
                           const aclScalar *scalar,
                           aclTensorList *out,
                           uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnForeachPowScalarV2,
                           void *workspace,
                           uint64_t workspaceSize,
                           aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnForeachRoundOffNumberV2GetWorkspaceSize,
                           const aclTensorList* x, const aclScalar* roundMode, aclTensorList* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnForeachRoundOffNumberV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnForeachSubListV2GetWorkspaceSize,
                           const aclTensorList* x1, const aclTensorList* x2, const aclScalar* alpha, aclTensorList* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnForeachSubListV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnForeachSubScalarV2GetWorkspaceSize,
                           const aclTensorList* x, const aclScalar* scalar, aclTensorList* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnForeachSubScalarV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnFracGetWorkspaceSize,
                           const aclTensor* input, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnFrac,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnFusedCrossEntropyLossWithMaxSumGetWorkspaceSize,
                           const aclTensor* logitsMax, const aclTensor* sumExpLogits,
                           const aclTensor* predictedLogits, float labelSmoothing, const aclTensor* inputOptional,
                           const aclTensor* weightOptional, const aclTensor* vocabParallelLogitsOptional, aclTensor* lossOut,
                           aclTensor* softMaxOutOptional, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnFusedCrossEntropyLossWithMaxSum,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize,
                           const aclTensor *grad, const aclTensor *input, const aclTensor *weight, const aclTensor *targetMask, const aclTensor *maskedTarget,
                           float labelSmoothing, const aclTensor *logitsMaxOptional, const aclTensor *sumExpLogitsOptional, const aclTensor *softmaxOptional,
                           aclTensor *inputGradOut, aclTensor *weightGradOut, uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnFusedLinearCrossEntropyLossGrad,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnFusedLinearOnlineMaxSumGetWorkspaceSize,
                           const aclTensor* input, const aclTensor* weight, const aclTensor* target, int64_t vocabStartIndex,
                           int64_t vocabEndIndex, aclTensor* logitsMaxLocalOut, aclTensor* sumExpLogitsLocalOut,
                           aclTensor* predictedLogitsLocalOut, aclTensor* targetMaskOut, aclTensor* maskedTargetOut,
                           aclTensor* vocabParallelLogitsOutOptional, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnFusedLinearOnlineMaxSum,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnFusedMatmulGetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x3, const char* fusedOpType,
                           int8_t cubeMathType, const aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnFusedMatmul,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnFusedQuantMatmulGetWorkspaceSize,
                           const aclTensor *x1, const aclTensor *x2,
                           const aclTensor *x1Scale, const aclTensor *x2Scale,
                           const aclTensor *yScaleOptional, const aclTensor *x1OffsetOptional,
                           const aclTensor *x2OffsetOptional, const aclTensor *yOffsetOptional,
                           const aclTensor *biasOptional, const aclTensor *x3Optional,
                           const char *fusedOpType, int64_t groupSizeOptional,
                           aclTensor *out, uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnFusedQuantMatmul,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnFusedQuantMatmulWeightNzGetWorkspaceSize,
                           const aclTensor *x1, const aclTensor *x2,
                           const aclTensor *x1Scale, const aclTensor *x2Scale,
                           const aclTensor *yScaleOptional, const aclTensor *x1OffsetOptional,
                           const aclTensor *x2OffsetOptional, const aclTensor *yOffsetOptional,
                           const aclTensor *biasOptional, const aclTensor *x3Optional,
                           const char *fusedOpType, int64_t groupSizeOptional,
                           aclTensor *out, uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnFusedQuantMatmulWeightNz,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGatherGetWorkspaceSize,
                           const aclTensor* self, const int64_t dim, const aclTensor* index, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGather,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGatherNdGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* indices,
                           bool negativeIndexSupport, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGatherNd,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGatherV2GetWorkspaceSize,
                           const aclTensor* self, int64_t dim, const aclTensor* index,
                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGatherV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGatherV3GetWorkspaceSize,
                           const aclTensor* self, int64_t dim, const aclTensor* index, int64_t batchDims,
                           int64_t mode, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGatherV3,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGcdGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGcd,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGeGluGetWorkspaceSize,
                           const aclTensor* self, int64_t dim, int64_t approximate, aclTensor* out, aclTensor* outGelu,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGeGlu,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGeGluBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gelu, int64_t dim, int64_t approximate,
                           aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGeGluBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGeGluV3GetWorkspaceSize,
                           const aclTensor* self, int64_t dim, int64_t approximate, bool activateLeft, aclTensor* out, aclTensor* outGelu,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGeGluV3,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGeGluV3BackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gelu, int64_t dim, int64_t approximate,
                           bool activateLeft, aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGeGluV3Backward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGeScalarGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* other, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGeScalar,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGeTensorGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* other, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGeTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGeluGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGelu,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGeluBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gradInput, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGeluBackward,
                           void* workspace, uint64_t workspace_size, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGeluBackwardV2GetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, char* approximate, aclTensor* gradInput,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGeluBackwardV2,
                           void* workspace, uint64_t workspace_size, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGeluQuantGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* inputScaleOptional, const aclTensor* inputOffsetOptional,
                           const char* approximate, const char* quantMode, const char* roundMode, int64_t dstType, const aclTensor* y,
                           const aclTensor* outScaleOptional, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGeluQuant,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGeluV2GetWorkspaceSize,
                           const aclTensor* x, int64_t approximate, aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGeluV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGemmGetWorkspaceSize,
                           const aclTensor* A, const aclTensor* B, const aclTensor* C, float alpha, float beta, int64_t transA, int64_t transB,
                           aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGemm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGerGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* vec2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGer,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGlobalAveragePoolGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGlobalAveragePool,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGlobalMaxPoolGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGlobalMaxPool,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGluGetWorkspaceSize,
                           const aclTensor* self, int64_t dim, const aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGlu,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGluBackwardGetWorkspaceSize,
                           const aclTensor* gradOut, const aclTensor* self, int64_t dim,
                           const aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGluBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGridSampler2DGetWorkspaceSize,
                           const aclTensor *input, const aclTensor *grid,
                           int64_t interpolationMode, int64_t paddingMode, bool alignCorners, aclTensor *out, uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGridSampler2D,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGridSampler2DBackwardGetWorkspaceSize,
                           const aclTensor *gradOutput, const aclTensor *input,
                           const aclTensor *grid, int64_t interpolationMode, int64_t paddingMode, bool alignCorners,
                           const aclBoolArray *outputMask, aclTensor *inputGrad, aclTensor *gridGrad, uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGridSampler2DBackward,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGridSampler3DGetWorkspaceSize,
                           const aclTensor *input, const aclTensor *grid,
                           int64_t interpolationMode, int64_t paddingMode, bool alignCorners, aclTensor *out, uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGridSampler3D,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGridSampler3DBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* input, const aclTensor* grid, int64_t interpolationMode,
                           int64_t paddingMode, bool alignCorners, const aclBoolArray* outputMask, aclTensor* inputGrad, aclTensor* gridGrad,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGridSampler3DBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupNormGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* gamma, const aclTensor* beta, int64_t N, int64_t C, int64_t HxW,
                           int64_t group, double eps, aclTensor* out, aclTensor* meanOut, aclTensor* rstdOut, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupNorm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupNormBackwardGetWorkspaceSize,
                           const aclTensor* gradOut, const aclTensor* input, const aclTensor* mean, const aclTensor* rstd,
                           const aclTensor* gamma, int64_t N, int64_t C, int64_t HxW, int64_t group, const aclBoolArray* outputMask,
                           aclTensor* gradInput, aclTensor* gradGammaOut, aclTensor* gradBetaOut, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupNormBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupNormSiluGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* gamma, const aclTensor* beta, int64_t group, double eps, aclTensor* out,
                           aclTensor* meanOut, aclTensor* rstdOut, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupNormSilu,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupNormSiluQuantGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* gamma, const aclTensor* beta, const aclTensor* quantScale, int64_t group, double eps, bool activateSilu,
                           aclTensor* out, aclTensor* meanOut, aclTensor* rstdOut, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupNormSiluQuant,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupNormSiluV2GetWorkspaceSize,
                           const aclTensor* self, const aclTensor* gamma, const aclTensor* beta, int64_t group, double eps, bool activateSilu,
                           aclTensor* out, aclTensor* meanOut, aclTensor* rstdOut, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupNormSiluV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupQuantGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* scale,
                           const aclTensor* groupIndex, const aclTensor* offsetOptional,
                           int32_t dstType, aclTensor* y,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupQuant,
                           void* workspace, uint64_t workspaceSize,
                           aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupedBiasAddGradGetWorkspaceSize,
                           const aclTensor* gradY, const aclTensor* groupIdxOptional, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupedBiasAddGrad,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupedBiasAddGradV2GetWorkspaceSize,
                           const aclTensor* gradY, const aclTensor* groupIdxOptional, int64_t groupIdxType, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupedBiasAddGradV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupedMatMulAllReduceGetWorkspaceSize,
                           const aclTensorList* x, const aclTensorList* weight, const aclTensorList* bias,
                           const aclIntArray* groupListOptional, int64_t splitItem, const char* group, const char* reduceOp, int64_t commTurn,
                           int64_t streamMode, const aclTensorList* y, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupedMatMulAllReduce,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupedMatMulAlltoAllvGetWorkspaceSize,
                           const aclTensor* gmmX, const aclTensor* gmmWeight, const aclTensor* sendCountsTensorOptional,
                           const aclTensor* recvCountsTensorOptional, const aclTensor* mmXOptional, const aclTensor* mmWeightOptional,
                           const char* group, int64_t epWorldSize, const aclIntArray* sendCounts, const aclIntArray* recvCounts,
                           bool transGmmWeight, bool transMmWeight, aclTensor* y, aclTensor* mmYOptional, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupedMatMulAlltoAllv,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupedMatmulFinalizeRoutingGetWorkspaceSize,
                           const aclTensor *x, aclTensor *w, const aclTensor *scaleOptional,
                           const aclTensor* biasOptional, const aclTensor *pertokenScaleOptional,
                           const aclTensor *groupListOptional, const aclTensor *sharedInputOptional, const aclTensor* logitOptional,
                           const aclTensor *rowIndexOptional, int64_t dtype, float sharedInputWeight,
                           int64_t sharedInputOffset, bool transposeX, bool transposeW, int64_t groupListType, aclTensor *y,
                           uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupedMatmulFinalizeRouting,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupedMatmulFinalizeRoutingV2GetWorkspaceSize,
                           const aclTensor *x1, aclTensor *x2,
                           const aclTensor *scaleOptional, const aclTensor *biasOptional,
                           const aclTensor *offsetOptional, const aclTensor *antiquantScaleOptional,
                           const aclTensor *antiquantOffsetOptional, const aclTensor *pertokenScaleOptional,
                           const aclTensor *groupListOptional, const aclTensor *sharedInputOptional,
                           const aclTensor *logitOptional, const aclTensor *rowIndexOptional, int64_t dtype,
                           float sharedInputWeight, int64_t sharedInputOffset, bool transposeX1, bool transposeX2,
                           int64_t groupListType, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupedMatmulFinalizeRoutingV2,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupedMatmulFinalizeRoutingV3GetWorkspaceSize,
                           const aclTensor *x1, aclTensor *x2,
                           const aclTensor *scaleOptional, const aclTensor *biasOptional,
                           const aclTensor *offsetOptional, const aclTensor *antiquantScaleOptional,
                           const aclTensor *antiquantOffsetOptional, const aclTensor *pertokenScaleOptional,
                           const aclTensor *groupListOptional, const aclTensor *sharedInputOptional,
                           const aclTensor *logitOptional, const aclTensor *rowIndexOptional, int64_t dtype,
                           float sharedInputWeight, int64_t sharedInputOffset, bool transposeX1, bool transposeX2,
                           int64_t groupListType, const aclIntArray *tuningConfigOptional, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupedMatmulFinalizeRoutingV3,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupedMatmulFinalizeRoutingWeightNzGetWorkspaceSize,
                           const aclTensor *x1, const aclTensor *x2, const aclTensor *scale,
                           const aclTensor* bias, const aclTensor *pertokenScaleOptional,
                           const aclTensor *groupList, const aclTensor *sharedInput, const aclTensor* logit,
                           const aclTensor *rowIndex, int64_t dtype, float sharedInputWeight,
                           int64_t sharedInputOffset, bool transposeX1, bool transposeX2, int64_t groupListType, aclTensor *out,
                           uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupedMatmulFinalizeRoutingWeightNz,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize,
                           const aclTensor *x1,
                           const aclTensor *x2, const aclTensor *scale, const aclTensor *bias, const aclTensor *offsetOptional,
                           const aclTensor *antiquantScaleOptional, const aclTensor *antiquantOffsetOptional,
                           const aclTensor *pertokenScaleOptional, const aclTensor *groupList, const aclTensor *sharedInput,
                           const aclTensor *logit, const aclTensor *rowIndex, int64_t dtype, float sharedInputWeight,
                           int64_t sharedInputOffset, bool transposeX1, bool transposeX2, int64_t groupListType,
                           const aclIntArray *tuningConfigOptional, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                           void *workspace, uint64_t workspaceSize,
                           aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnHardshrinkGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* lambd, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnHardshrink,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnHardshrinkBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, const aclScalar* lambd, aclTensor* gradInput,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnHardshrinkBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnHardsigmoidGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnHardsigmoid,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnHardsigmoidBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnHardsigmoidBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnHardswishGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnHardswish,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnHardswishBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnHardswishBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnHardswishBackwardV2GetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnHardswishBackwardV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnHardtanhGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* clipValueMin, const aclScalar* clipValueMax, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnHardtanh,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnHardtanhBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self,
                           const aclScalar* min, const aclScalar* max, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnHardtanhBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnHeavisideGetWorkspaceSize,
                           const aclTensor* input, const aclTensor* values, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnHeaviside,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnHistcGetWorkspaceSize,
                           const aclTensor* self, int64_t bins, const aclScalar* min, const aclScalar* max, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnHistc,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnIm2colGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* dilation, const aclIntArray* padding,
                           const aclIntArray* stride, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnIm2col,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnIm2colBackwardGetWorkspaceSize,
                           const aclTensor *gradOutput, const aclIntArray *inputSize,
                           const aclIntArray *kernelSize, const aclIntArray *dilation, const aclIntArray *padding, const aclIntArray *stride,
                           aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnIm2colBackward,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnIndexGetWorkspaceSize,
                           const aclTensor* self, const aclTensorList* indices, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnIndex,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnIndexAddGetWorkspaceSize,
                           const aclTensor* self, const int64_t dim, const aclTensor* index,
                           const aclTensor* source, const aclScalar* alpha, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnIndexAdd,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnIndexAddV2GetWorkspaceSize,
                           const aclTensor* self, const int64_t dim, const aclTensor* index,
                           const aclTensor* source, const aclScalar* alpha, int64_t mode, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnIndexAddV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnIndexCopyGetWorkspaceSize,
                           aclTensor* selfRef, int64_t dim, const aclTensor* index,
                           const aclTensor* source, aclTensor* outRef,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnIndexCopy,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnIndexFillGetWorkspaceSize,
                           const aclTensor* self, int64_t dim, const aclTensor* index, const aclScalar* value, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnIndexFill,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnIndexFillTensorGetWorkspaceSize,
                           const aclTensor* self, int64_t dim, const aclIntArray* index,
                           const aclScalar* value, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnIndexFillTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnIndexPutImplGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensorList* indices,
                           const aclTensor* values, const bool accumulate,
                           const bool unsafe, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnIndexPutImpl,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnIndexSelectGetWorkspaceSize,
                           const aclTensor* self, int64_t dim, const aclTensor* index,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnIndexSelect,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAcosGetWorkspaceSize,
                           aclTensor* inputRef, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAcos,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAcoshGetWorkspaceSize,
                           aclTensor* selfRef, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAcosh,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAddGetWorkspaceSize,
                           const aclTensor* selfRef, const aclTensor* other, const aclScalar* alpha, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAdd,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAddReluGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* other, aclScalar* alpha,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAddRelu,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAddV3GetWorkspaceSize,
                           const aclScalar* selfRef, const aclTensor* other, const aclScalar* alpha, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAddV3,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAddbmmGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* batch1, const aclTensor* batch2, const aclScalar* beta, const aclScalar* alpha,
                           int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAddbmm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAddcdivGetWorkspaceSize,
                           const aclTensor* selfRef, const aclTensor* tensor1, const aclTensor* tensor2, const aclScalar* value,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAddcdiv,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAddcmulGetWorkspaceSize,
                           const aclTensor* selfRef, const aclTensor* tensor1, const aclTensor* tensor2, const aclScalar* value,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAddcmul,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAddmmGetWorkspaceSize,
                           const aclTensor* selfRef, const aclTensor* mat1, const aclTensor* mat2, const aclScalar* beta,
                           const aclScalar* alpha, int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAddmm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAddrGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* vec1, const aclTensor* vec2, const aclScalar* betaOptional,
                           const aclScalar* alphaOptional, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAddr,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAddsGetWorkspaceSize,
                           const aclTensor* selfRef, const aclScalar* other, const aclScalar* alpha, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAdds,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAsinGetWorkspaceSize,
                           aclTensor* selfRef, uint64_t* workspace_size,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAsin,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAsinhGetWorkspaceSize,
                           aclTensor* inputRef, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAsinh,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAtanGetWorkspaceSize,
                           aclTensor* inputRef, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAtan,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAtan2GetWorkspaceSize,
                           aclTensor* selfRef, aclTensor* other, uint64_t* workspace_size,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAtan2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAtanhGetWorkspaceSize,
                           aclTensor* inputRef, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAtanh,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAttentionWorkerSchedulerGetWorkspaceSize,
                           aclTensor *scheduleContextRef,
                           uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAttentionWorkerScheduler,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceBaddbmmGetWorkspaceSize,
                           const aclTensor* selfRef, const aclTensor* batch1, const aclTensor* batch2, const aclScalar* beta,
                           const aclScalar* alpha, int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceBaddbmm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceBernoulliGetWorkspaceSize,
                           const aclTensor* selfRef, const aclScalar* prob, int64_t seed, int64_t offset, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceBernoulli,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceBernoulliTensorGetWorkspaceSize,
                           const aclTensor* selfRef, const aclTensor* prob, int64_t seed, int64_t offset, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceBernoulliTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceBitwiseAndScalarGetWorkspaceSize,
                           const aclTensor* selfRef, const aclScalar* other,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceBitwiseAndScalar,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceBitwiseAndTensorGetWorkspaceSize,
                           const aclTensor* selfRef, const aclTensor* other,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceBitwiseAndTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceBitwiseAndTensorOutGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* other,
                           uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceBitwiseAndTensorOut,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceBitwiseOrScalarGetWorkspaceSize,
                           aclTensor* selfRef, const aclScalar* other,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceBitwiseOrScalar,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceBitwiseOrTensorGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* other,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceBitwiseOrTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceBitwiseXorScalarGetWorkspaceSize,
                           aclTensor* selfRef, const aclScalar* other,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceBitwiseXorScalar,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceBitwiseXorTensorGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* other,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceBitwiseXorTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceCeilGetWorkspaceSize,
                           aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceCeil,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceCeluGetWorkspaceSize,
                           aclTensor* selfRef, const aclScalar* alpha, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceCelu,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceClampMaxGetWorkspaceSize,
                           const aclTensor* selfRef, const aclScalar* clipValueMax, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceClampMax,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceClampMaxTensorGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* max, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceClampMaxTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceClampMinTensorGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* clipValueMin, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceClampMinTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceCosGetWorkspaceSize,
                           aclTensor* inputRef, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceCos,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceCoshGetWorkspaceSize,
                           aclTensor* selfRef, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceCosh,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceCumprodGetWorkspaceSize,
                           aclTensor *input, const aclScalar *dim, uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceCumprod,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceDivGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceDiv,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceDivModGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* other, int mode, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceDivMod,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceDivModsGetWorkspaceSize,
                           aclTensor* selfRef, const aclScalar* other, int mode, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceDivMods,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceDivsGetWorkspaceSize,
                           aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceDivs,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceEluGetWorkspaceSize,
                           aclTensor* selfRef, const aclScalar* alpha, const aclScalar* scale, const aclScalar* inputScale,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceElu,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceEqScalarGetWorkspaceSize,
                           const aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceEqScalar,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceEqTensorGetWorkspaceSize,
                           const aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceEqTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceErfGetWorkspaceSize,
                           aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceErf,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceErfcGetWorkspaceSize,
                           const aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceErfc,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceErfinvGetWorkspaceSize,
                           const aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceErfinv,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceExpGetWorkspaceSize,
                           aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceExp,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceExp2GetWorkspaceSize,
                           aclTensor* selfRef, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceExp2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceExpm1GetWorkspaceSize,
                           aclTensor* selfRef, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceExpm1,
                           void* workspace, uint64_t workspace_size, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceFfnWorkerSchedulerGetWorkspaceSize,
                           aclTensor* scheduleContextRef, int32_t syncGroupSize,
                           int32_t executeMode, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceFfnWorkerScheduler,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceFloorGetWorkspaceSize,
                           aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceFloor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceFloorDivideGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceFloorDivide,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceFloorDividesGetWorkspaceSize,
                           aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceFloorDivides,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceFmodScalarGetWorkspaceSize,
                           aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceFmodScalar,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceFmodTensorGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceFmodTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceFracGetWorkspaceSize,
                           aclTensor* inputRef, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceFrac,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceGeScalarGetWorkspaceSize,
                           aclTensor* selfRef, const aclScalar* other,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceGeScalar,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceGeTensorGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* other,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceGeTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceGtScalarGetWorkspaceSize,
                           aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceGtScalar,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceGtTensorGetWorkspaceSize,
                           const aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceGtTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceHardsigmoidGetWorkspaceSize,
                           const aclTensor* self, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceHardsigmoid,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceHardswishGetWorkspaceSize,
                           const aclTensor* self, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceHardswish,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceHardtanhGetWorkspaceSize,
                           aclTensor* selfRef, const aclScalar* clipValueMin, const aclScalar* clipValueMax, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceHardtanh,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceIndexCopyGetWorkspaceSize,
                           aclTensor* selfRef, int64_t dim, const aclTensor* index,
                           const aclTensor* source, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceIndexCopy,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceIndexFillGetWorkspaceSize,
                           aclTensor* selfRef, int64_t dim, const aclTensor* index, const aclScalar* value, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceIndexFill,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceIndexFillTensorGetWorkspaceSize,
                           aclTensor* selfRef, int64_t dim,
                           const aclIntArray* index, const aclScalar* value,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceIndexFillTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLeScalarGetWorkspaceSize,
                           aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLeScalar,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLeTensorGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLeTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLeakyReluGetWorkspaceSize,
                           aclTensor* selfRef, const aclScalar* negativeSlope,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLeakyRelu,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLerpGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* end,
                           const aclTensor* weight, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLerp,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLerpsGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* end,
                           const aclScalar* weight, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLerps,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLogGetWorkspaceSize,
                           aclTensor* selfRef, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLog,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLog10GetWorkspaceSize,
                           aclTensor* selfRef, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLog10,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLog1pGetWorkspaceSize,
                           aclTensor* selfRef, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLog1p,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLog2GetWorkspaceSize,
                           const aclTensor* selfRef, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLog2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLogicalAndGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLogicalAnd,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLogicalNotGetWorkspaceSize,
                           aclTensor* selfRef, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLogicalNot,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLogicalOrGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* other,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLogicalOr,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLtScalarGetWorkspaceSize,
                           const aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLtScalar,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLtTensorGetWorkspaceSize,
                           const aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLtTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceMaskedFillScalarGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* mask,
                           const aclScalar* value, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceMaskedFillScalar,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceMaskedFillTensorGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* mask, const aclTensor* value, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceMaskedFillTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceMaskedScatterGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* mask, const aclTensor* source, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceMaskedScatter,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceMatmulAllReduceAddRmsNormGetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* residual, const aclTensor* gamma,
                           double epsilon, const char* group, const char* reduceOp, int64_t commTurn, int64_t streamMode,
                           const aclTensor* normOut, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceMatmulAllReduceAddRmsNorm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceMishGetWorkspaceSize,
                           aclTensor* selfRef, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceMish,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceMulGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* other,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceMul,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceMulsGetWorkspaceSize,
                           aclTensor* selfRef, const aclScalar* other,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceMuls,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceNanToNumGetWorkspaceSize,
                           aclTensor* selfRef, float nan, float posinf, float neginf,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceNanToNum,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceNeScalarGetWorkspaceSize,
                           aclTensor* selfRef, const aclScalar* other,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceNeScalar,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceNeTensorGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* other,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceNeTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceNegGetWorkspaceSize,
                           aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceNeg,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplacePowTensorScalarGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* exponent,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplacePowTensorScalar,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplacePowTensorTensorGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* exponent,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplacePowTensorTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplacePutGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* index,
                           const aclTensor* source, bool accumulate, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplacePut,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceQuantMatmulAllReduceAddRmsNormGetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* dequantScale,
                           const aclTensor* residual, const aclTensor* gamma, double epsilon, const char* group, const char* reduceOp,
                           int64_t commTurn, int64_t streamMode, const aclTensor* normOut, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceQuantMatmulAllReduceAddRmsNorm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceQuantScatterGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* indices,
                           const aclTensor* updates, const aclTensor* quantScales,
                           const aclTensor* quantZeroPoints, int64_t axis,
                           int64_t quantAxis, int64_t reduction,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceQuantScatter,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceQuantScatterV2GetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* indices,
                           const aclTensor* updates, const aclTensor* quantScales,
                           const aclTensor* quantZeroPoints, int64_t axis,
                           int64_t quantAxis, int64_t reduction, const char* roundMode,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceQuantScatterV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceRReluWithNoiseGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* noise,
                           const aclScalar* lower, const aclScalar* upper,
                           bool training, int64_t seed, int64_t offset,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceRReluWithNoise,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceReciprocalGetWorkspaceSize,
                           const aclTensor* selfRef, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceReciprocal,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceReluGetWorkspaceSize,
                           aclTensor* selfRef, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceRelu,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceRemainderTensorScalarGetWorkspaceSize,
                           aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceRemainderTensorScalar,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceRemainderTensorTensorGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceRemainderTensorTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceRenormGetWorkspaceSize,
                           aclTensor* selfRef, const aclScalar* p, int64_t dim, const aclScalar* maxNorm, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceRenorm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceRoundGetWorkspaceSize,
                           const aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceRound,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceRoundDecimalsGetWorkspaceSize,
                           aclTensor* selfRef, int64_t decimals, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceRoundDecimals,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceRsqrtGetWorkspaceSize,
                           aclTensor* selfRef, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceRsqrt,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceScatterGetWorkspaceSize,
                           aclTensor* selfRef, int64_t dim, const aclTensor* index, const aclTensor* src, int64_t reduce,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceScatter,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceScatterUpdateGetWorkspaceSize,
                           aclTensor* data, const aclTensor* indices,
                           const aclTensor* updates, int64_t axis,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceScatterUpdate,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceScatterValueGetWorkspaceSize,
                           aclTensor* selfRef, int64_t dim, const aclTensor* index, const aclScalar* value, int64_t reduce,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceScatterValue,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceSeluGetWorkspaceSize,
                           aclTensor* selfRef, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceSelu,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceSigmoidGetWorkspaceSize,
                           aclTensor* selfRef, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceSigmoid,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceSinGetWorkspaceSize,
                           aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceSin,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceSincGetWorkspaceSize,
                           aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceSinc,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceSinhGetWorkspaceSize,
                           aclTensor* selfRef, uint64_t* workspace_size,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceSinh,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceSqrtGetWorkspaceSize,
                           aclTensor* self, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceSqrt,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceSubGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* other, const aclScalar* alpha, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceSub,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceSubsGetWorkspaceSize,
                           aclTensor* selfRef, const aclScalar* other, const aclScalar* alpha, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceSubs,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceTanGetWorkspaceSize,
                           const aclTensor* selfRef, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceTan,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceTanhGetWorkspaceSize,
                           aclTensor* selfRef, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceTanh,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceThresholdGetWorkspaceSize,
                           aclTensor* selfRef, const aclScalar* threshold, const aclScalar* value, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceThreshold,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceTrilGetWorkspaceSize,
                           const aclTensor* selfRef, int64_t diagonal, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceTril,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceTriuGetWorkspaceSize,
                           aclTensor* selfRef, int64_t diagonal, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceTriu,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceTruncGetWorkspaceSize,
                           aclTensor* selfRef, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceTrunc,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* antiquantScale,
                           const aclTensor* antiquantOffset, const aclTensor* residual, const aclTensor* gamma, double epsilon,
                           const char* group, const char* reduceOp, int64_t commTurn, int64_t streamMode, int64_t antiquantGroupSize,
                           const aclTensor* normOut, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceXLogYScalarOtherGetWorkspaceSize,
                           aclTensor* selfRef, const aclScalar* other,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceXLogYScalarOther,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceXLogYTensorGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* other,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceXLogYTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInstanceNormGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* gamma, const aclTensor* beta, const char* dataFormat, double eps, aclTensor* y,
                           aclTensor* mean, aclTensor* variance, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInstanceNorm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInterleaveRopeGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* cos, const aclTensor* sin, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInterleaveRope,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInverseGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInverse,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnIouGetWorkspaceSize,
                           const aclTensor *bBoxes, const aclTensor *gtBoxes,
                           const char *mode, float eps, bool aligned,
                           aclTensor *overlap, uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnIou,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnIsCloseGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* other, double rtol, double atol, bool equal_nan, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnIsClose,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnIsInScalarTensorGetWorkspaceSize,
                           const aclScalar* element, const aclTensor* testElements, bool assumeUnique, bool invert, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnIsInScalarTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnIsInTensorScalarGetWorkspaceSize,
                           const aclTensor* element, const aclScalar* testElement, bool assumeUnique, bool invert, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnIsInTensorScalar,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnIsNegInfGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnIsNegInf,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnIsPosInfGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnIsPosInf,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnKlDivGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* target, int64_t reduction,
                           bool logTarget, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnKlDiv,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnKlDivBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, int64_t reduction, bool logTarget,
                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnKlDivBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnKlDivTargetBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, int64_t reduction, bool logTarget,
                           aclTensor* gradTarget, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnKlDivTargetBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnKthvalueGetWorkspaceSize,
                           const aclTensor* self, int64_t k, int64_t dim, bool keepdim,
                           aclTensor* valuesOut, aclTensor* indicesOut,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnKthvalue,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnL1LossGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* target, int64_t reduction, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnL1Loss,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnL1LossBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, int64_t reduction,
                           aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnL1LossBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLSTMGetWorkspaceSize,
                           const aclTensor *input,
                           const aclTensorList *params,
                           const aclTensorList *hx,
                           const aclTensor *batchSizes,
                           bool has_biases,
                           int64_t numLayers,
                           double droupout,
                           bool train,
                           bool bidirectional,
                           bool batch_first,
                           aclTensor *output,
                           aclTensor *hy,
                           aclTensor *cy,
                           aclTensorList *iOut,
                           aclTensorList *jOut,
                           aclTensorList *fOut,
                           aclTensorList *oOut,
                           aclTensorList *hOut,
                           aclTensorList *cOut,
                           aclTensorList *tanhCOut,
                           uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLSTM,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLayerNormGetWorkspaceSize,
                           const aclTensor* input, const aclIntArray* normalizedShape, const aclTensor* weightOptional,
                           const aclTensor* biasOptional, double eps, aclTensor* out, aclTensor* meanOutOptional, aclTensor* rstdOutOptional,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLayerNorm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLayerNormBackwardGetWorkspaceSize,
                           const aclTensor* gradOut, const aclTensor* input, const aclIntArray* normalizedShape, const aclTensor* mean,
                           const aclTensor* rstd, const aclTensor* weightOptional, const aclTensor* biasOptional,
                           const aclBoolArray* outputMask, aclTensor* gradInputOut, aclTensor* gradWeightOut, aclTensor* gradBiasOut,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLayerNormBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLayerNormQuantGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* gammma, const aclTensor* beta, const aclTensor* scale,
                           const aclTensor* zeroPointsOptional, int quantMode, double epsilon, aclTensor* res, aclTensor* scaleOut,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLayerNormQuant,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLayerNormWithImplModeGetWorkspaceSize,
                           const aclTensor* input, const aclIntArray* normalizedShape, const aclTensor* weightOptional,
                           const aclTensor* biasOptional, double eps, aclTensor* out, aclTensor* meanOutOptional, aclTensor* rstdOutOptional,
                           int32_t implMode, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLayerNormWithImplMode,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLeScalarGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLeScalar,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLeTensorGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLeTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLeakyReluGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* negativeSlope,
                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLeakyRelu,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLeakyReluBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, const aclScalar* negativeSlope, bool selfIsResult,
                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLeakyReluBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLeftShiftGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* shiftBits, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLeftShift,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLeftShiftsGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* shiftBits, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLeftShifts,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLerpGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* end, const aclTensor* weight,
                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLerp,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLerpsGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* end, const aclScalar* weight,
                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLerps,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLgammaGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLgamma,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLinalgCholeskyGetWorkspaceSize,
                           const aclTensor* self, bool upper, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLinalgCholesky,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLinalgCrossGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* other, int64_t dim,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLinalgCross,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLinalgQrGetWorkspaceSize,
                           const aclTensor* self, int64_t mode, aclTensor* Q, aclTensor* R,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLinalgQr,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLinalgVectorNormGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* ord, const aclIntArray* dims, bool keepDims, const aclDataType dtype,
                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLinalgVectorNorm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLinspaceGetWorkspaceSize,
                           const aclScalar* start, const aclScalar* end, int64_t steps,
                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLinspace,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLog,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLog10GetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLog10,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLog1pGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLog1p,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLog2GetWorkspaceSize,
                           const aclTensor* self, const aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLog2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogAddExpGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* other, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogAddExp,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogAddExp2GetWorkspaceSize,
                           const aclTensor* self, const aclTensor* other, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogAddExp2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogSigmoidGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogSigmoid,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogSigmoidBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self,
                           const aclTensor* buffer, aclTensor* gradInput,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogSigmoidBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogSigmoidForwardGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, aclTensor* buffer,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogSigmoidForward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogSoftmaxGetWorkspaceSize,
                           const aclTensor* self, int64_t dim, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogSoftmax,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogSoftmaxBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* output,
                           int64_t dim, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogSoftmaxBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogSpaceGetWorkspaceSize,
                           const aclScalar *start, const aclScalar *end, int64_t steps, double base,
                           const aclTensor *result, uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogSpace,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogdetGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogdet,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogicalAndGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogicalAnd,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogicalNotGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogicalNot,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogicalOrGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* other, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogicalOr,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogicalXorGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* other, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLogicalXor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnLstmBackwardGetWorkspaceSize,
                           const aclTensor *input,
                           const aclTensorList *hx,
                           const aclTensorList *params,
                           const aclTensor *dy,
                           const aclTensor *dh,
                           const aclTensor *dc,
                           const aclTensorList *i,
                           const aclTensorList *g,
                           const aclTensorList *f,
                           const aclTensorList *o,
                           const aclTensorList *h,
                           const aclTensorList *c,
                           const aclTensorList *tanhc,
                           const aclTensor *batchSizesOptional,
                           bool hasBias,
                           int64_t numLayers,
                           double dropout,
                           bool train,
                           bool bidirectional,
                           bool batchFirst,
                           const aclBoolArray *outputMask,
                           aclTensor *dxOut,
                           aclTensor *dhPrevOut,
                           aclTensor *dcPrevOut,
                           aclTensorList *dparamsOut,
                           uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnLstmBackward,
                           void *workspace,
                           uint64_t workspaceSize,
                           aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMaskedScaleGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* mask, float scale,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMaskedScale,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMatmulGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* mat2, aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMatmul,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMatmulAllReduceGetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const char* group, const char* reduceOp,
                           int64_t commTurn, int64_t streamMode, const aclTensor* output, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMatmulAllReduce,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMatmulAllReduceAddRmsNormGetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* residual, const aclTensor* gamma,
                           double epsilon, const char* group, const char* reduceOp, int64_t commTurn, int64_t streamMode, const aclTensor* y,
                           const aclTensor* normOut, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMatmulAllReduceAddRmsNorm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMatmulAllReduceV2GetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x3, const char* group,
                           const char* reduceOp, int64_t commTurn, int64_t streamMode, const aclTensor* output, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMatmulAllReduceV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMatmulCompressGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* weight,
                           const aclTensor* bias, const aclTensor* compressIndex,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMatmulCompress,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMatmulCompressDequantGetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2,
                           const aclTensor* compressIndex, const aclTensor* bias,
                           const aclTensor* deqScale, const aclTensor* offsetW,
                           int offsetX, const aclIntArray* compressInfo,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMatmulCompressDequant,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMatmulReduceScatterGetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2,
                           const aclTensor* bias, const char* group,
                           const char* reduceOp, int64_t commTurn,
                           int64_t streamMode, const aclTensor* output,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMatmulReduceScatter,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMatmulReduceScatterV2GetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2,
                           const aclTensor* bias, const aclTensor* x1Scale,
                           const aclTensor* x2Scale, const aclTensor* quantScale,
                           int64_t blockSize, const char* group,
                           const char* reduceOp, int64_t commTurn,
                           int64_t streamMode, int64_t groupSize, const char* commMode,
                           aclTensor* output, aclTensor* amaxOutOptional,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMatmulReduceScatterV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMatmulWeightNzGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* mat2, aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMatmulWeightNz,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMaxPoolGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* kernelShape,
                           const aclIntArray* strides, const int64_t autoPad,
                           const aclIntArray* pads, const aclIntArray* dilations,
                           const int64_t ceilMode, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMaxPool,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMaxPool2dWithIndicesGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding,
                           const aclIntArray* dilation, bool ceilMode, aclTensor* out, aclTensor* indices, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMaxPool2dWithIndices,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const aclIntArray* kernelSize,
                           const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation, bool ceilMode,
                           aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMaxPool2dWithIndicesBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMaxPool2dWithMaskGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding,
                           const aclIntArray* dilation, bool ceilMode, aclTensor* out, aclTensor* indices, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMaxPool2dWithMask,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const aclIntArray* kernelSize,
                           const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation, bool ceilMode,
                           aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMaxPool2dWithMaskBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMaxPool3dWithArgmaxGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding,
                           const aclIntArray* dilation, bool ceilMode, aclTensor* out, aclTensor* indices, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMaxPool3dWithArgmax,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMaxPool3dWithArgmaxBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const aclIntArray* kernelSize,
                           const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation, bool ceilMode,
                           aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMaxPool3dWithArgmaxBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMaxUnpool2dGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* indices,
                           const aclIntArray* outputSize, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMaxUnpool2d,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMaxUnpool2dBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self,
                           const aclTensor* indices, const aclIntArray* outputSize,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMaxUnpool2dBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMaxUnpool3dGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* indices,
                           const aclIntArray* outputSize, const aclIntArray* stride,
                           const aclIntArray* padding, aclTensor* outRef,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMaxUnpool3d,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMaxUnpool3dBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self,
                           const aclTensor* indices, const aclIntArray* outputSize,
                           const aclIntArray* stride, const aclIntArray* padding,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMaxUnpool3dBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMedianGetWorkspaceSize,
                           const aclTensor* self, aclTensor* valuesOut, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMedian,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMedianDimGetWorkspaceSize,
                           const aclTensor* self, int64_t dim, bool keepDim,
                           aclTensor* valuesOut, aclTensor* indicesOut,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMedianDim,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMishGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMish,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMishBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self,
                           aclTensor* gradInput, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMishBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMmGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* mat2, aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnModulateGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* scaleOptional, const aclTensor* shiftOptional, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnModulate,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnModulateBackwardGetWorkspaceSize,
                           const aclTensor* grad_output, const aclTensor* input,const aclTensor* scale,const aclTensor* shift,
                           const aclTensor* grad_input,const aclTensor* grad_scale,const aclTensor* grad_shift,uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnModulateBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeCombineGetWorkspaceSize,
                           const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *expandIdx, const aclTensor *epSendCounts,
                           const aclTensor *expertScales, const aclTensor *tpSendCounts, const aclTensor *xActiveMask,
                           const aclTensor *activationScale, const aclTensor *weightScale, const aclTensor *groupList,
                           const aclTensor *expandScales, const char *groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
                           const char *groupTp, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
                           int64_t sharedExpertRankNum, int64_t globalBs, int64_t outDtype, int64_t commQuantMode, int64_t groupListType,
                           aclTensor *x, uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeCombine,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSize,
                           const aclTensor* expandX, const aclTensor* expertIds,
                           const aclTensor* assistInfoForCombine, const aclTensor* epSendCounts,
                           const aclTensor* expertScales, const aclTensor* residualX,
                           const aclTensor* gamma, const aclTensor* tpSendCountsOptional,
                           const aclTensor* xActiveMaskOptional, const aclTensor* activationScaleOptional,
                           const aclTensor* weightScaleOptional, const aclTensor* groupListOptional,
                           const aclTensor* expandScalesOptional,  const aclTensor* sharedExpertXOptional,
                           const char* groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
                           const char* groupTp, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType,
                           int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t globalBs, int64_t outDtype,
                           int64_t commQuantMode, int64_t groupListType, const char* commAlg, float normEps,
                           aclTensor* yOut, aclTensor* rstdOut, aclTensor* xOut, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeCombineAddRmsNorm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeCombineAddRmsNormV2GetWorkspaceSize,
                           const aclTensor* expandX, const aclTensor* expertIds,
                           const aclTensor* assistInfoForCombine, const aclTensor* epSendCounts,
                           const aclTensor* expertScales, const aclTensor* residualX,
                           const aclTensor* gamma, const aclTensor* tpSendCountsOptional,
                           const aclTensor* xActiveMaskOptional, const aclTensor* activationScaleOptional,
                           const aclTensor* weightScaleOptional, const aclTensor* groupListOptional,
                           const aclTensor* expandScalesOptional,  const aclTensor* sharedExpertXOptional,
                           const aclTensor* elasticInfoOptional, const aclTensor* oriXOptional,
                           const aclTensor* constExpertAlpha1Optional, const aclTensor* constExpertAlpha2Optional,
                           const aclTensor* constExpertVOptional, const char* groupEp, int64_t epWorldSize,
                           int64_t epRankId, int64_t moeExpertNum, const char* groupTp, int64_t tpWorldSize,
                           int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
                           int64_t sharedExpertRankNum, int64_t globalBs, int64_t outDtype,
                           int64_t commQuantMode, int64_t groupListType, const char* commAlg, float normEps,
                           int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum,
                           aclTensor* yOut, aclTensor* rstdOut, aclTensor* xOut, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeCombineAddRmsNormV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeCombineSetupGetWorkspaceSize,
                           const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *assistInfoForCombine, const char *groupEp,
                           int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, int64_t expertSharedType, int64_t sharedExpertNum,
                           int64_t sharedExpertRankNum, int64_t globalBs, int64_t commQuantMode, int64_t commType, const char *commAlg,
                           aclTensor *quantExpandXOut, aclTensor *commCmdInfoOut, uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeCombineSetup,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeCombineSetupTeardownCalcOutputSize,
                           const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *assistInfoForCombine, const char *groupEp,
                           int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, int64_t expertSharedType, int64_t sharedExpertNum,
                           int64_t sharedExpertRankNum, int64_t globalBs, int64_t commQuantMode, int64_t commType, const char *commAlg,
                           uint64_t &tokenMsgSize, uint64_t &commCmdInfoOutSize)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeCombineTeardownGetWorkspaceSize,
                           const aclTensor *expandX, const aclTensor *quantExpandX, const aclTensor *expertIds, const aclTensor *expandIdx,
                           const aclTensor *expertScales, const aclTensor *commCmdInfo, const aclTensor *xActiveMaskOptional,
                           const aclTensor *sharedExpertXOptional, const char *groupEp, int64_t epWorldSize, int64_t epRankId,
                           int64_t moeExpertNum, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
                           int64_t globalBs, int64_t commQuantMode, int64_t commType, const char *commAlg, aclTensor *xOut,
                           uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeCombineTeardown,
                           void *workspace, uint64_t workspaceSize,
                           aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeCombineV2GetWorkspaceSize,
                           const aclTensor* expandX, const aclTensor* expertIds,
                           const aclTensor* assistInfoForCombine, const aclTensor* epSendCounts,
                           const aclTensor* expertScales, const aclTensor* tpSendCountsOptional,
                           const aclTensor* xActiveMaskOptional, const aclTensor* activationScaleOptional,
                           const aclTensor* weightScaleOptional, const aclTensor* groupListOptional, const aclTensor* expandScalesOptional,
                           const aclTensor* sharedExpertXOptional,
                           const char* groupEp, int64_t epWorldSize,
                           int64_t epRankId, int64_t moeExpertNum,
                           const char* groupTp, int64_t tpWorldSize, int64_t tpRankId,
                           int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
                           int64_t globalBs, int64_t outDtype, int64_t commQuantMode,
                           int64_t groupListType, const char* commAlg, aclTensor* xOut, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeCombineV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeCombineV3GetWorkspaceSize,
                           const aclTensor* expandX, const aclTensor* expertIds,
                           const aclTensor* assistInfoForCombine, const aclTensor* epSendCounts,
                           const aclTensor* expertScales, const aclTensor* tpSendCountsOptional,
                           const aclTensor* xActiveMaskOptional, const aclTensor* activationScaleOptional,
                           const aclTensor* weightScaleOptional, const aclTensor* groupListOptional, const aclTensor* expandScalesOptional,
                           const aclTensor* sharedExpertXOptional, const aclTensor* elasticInfoOptional, const aclTensor* oriXOptional,
                           const aclTensor* constExpertAlpha1Optional, const aclTensor* constExpertAlpha2Optional,
                           const aclTensor* constExpertVOptional,
                           const char* groupEp, int64_t epWorldSize,
                           int64_t epRankId, int64_t moeExpertNum,
                           const char* groupTp, int64_t tpWorldSize, int64_t tpRankId,
                           int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
                           int64_t globalBs, int64_t outDtype, int64_t commQuantMode,
                           int64_t groupListType, const char* commAlg, int64_t zeroExpertNum,
                           int64_t copyExpertNum, int64_t constExpertNum,
                           aclTensor* xOut, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeCombineV3,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeCombineV4GetWorkspaceSize,
                           const aclTensor* expandX, const aclTensor* expertIds,
                           const aclTensor* assistInfoForCombine, const aclTensor* epSendCounts,
                           const aclTensor* expertScales, const aclTensor* tpSendCountsOptional,
                           const aclTensor* xActiveMaskOptional, const aclTensor* activationScaleOptional,
                           const aclTensor* weightScaleOptional, const aclTensor* groupListOptional, const aclTensor* expandScalesOptional,
                           const aclTensor* sharedExpertXOptional, const aclTensor* elasticInfoOptional, const aclTensor* oriXOptional,
                           const aclTensor* constExpertAlpha1Optional, const aclTensor* constExpertAlpha2Optional,
                           const aclTensor* constExpertVOptional, const aclTensor* performanceInfoOptional,
                           const char* groupEp, int64_t epWorldSize,
                           int64_t epRankId, int64_t moeExpertNum,
                           const char* groupTp, int64_t tpWorldSize, int64_t tpRankId,
                           int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
                           int64_t globalBs, int64_t outDtype, int64_t commQuantMode,
                           int64_t groupListType, const char* commAlg, int64_t zeroExpertNum,
                           int64_t copyExpertNum, int64_t constExpertNum,
                           aclTensor* xOut, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeCombineV4,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeDispatchGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* expertIds,
                           const aclTensor* scales, const aclTensor* xActiveMask,
                           const aclTensor* expertScales,
                           const char* groupEp, int64_t epWorldSize, int64_t epRankId,
                           int64_t moeExpertNum, const char* groupTp, int64_t tpWorldSize,
                           int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
                           int64_t sharedExpertRankNum, int64_t quantMode, int64_t globalBs,
                           int64_t expertTokenNumsType,
                           aclTensor* expandX, aclTensor* dynamicScales,
                           aclTensor* expandIdx, aclTensor* expertTokenNums,
                           aclTensor* epRecvCounts, aclTensor* tpRecvCounts,
                           aclTensor* expandScales, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeDispatch,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeDispatchSetupGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* expertIds, const aclTensor* scalesOptional,
                           const aclTensor* xActiveMaskOptional, const char* groupEp, int64_t epWorldSize, int64_t epRankId,
                           int64_t moeExpertNum, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
                           int64_t quantMode, int64_t globalBs, int64_t commType, const char* commAlg, aclTensor* yOut,
                           aclTensor* expandIdxOut, aclTensor* commCmdInfoOut, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeDispatchSetup,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize,
                           const aclTensor* x, const aclTensor* expertIds, const aclTensor* scalesOptional,
                           const aclTensor* xActiveMaskOptional, const char* groupEp, int64_t epWorldSize, int64_t epRankId,
                           int64_t moeExpertNum, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
                           int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, int64_t commType, const char* commAlg,
                           uint64_t& tokenMsgSize, uint64_t& expandIdxOutSize, uint64_t& assistInfoForCombineOutSize,
                           uint64_t& commCmdInfoOutSize)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeDispatchTeardownGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* y, const aclTensor* expertIds, const aclTensor* commCmdInfo,
                           const char* groupEp, int64_t epWorldSize, int64_t epRankId,
                           int64_t moeExpertNum, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
                           int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, int64_t commType, char* commAlg,
                           aclTensor* expandXOut, aclTensor* dynamicScalesOut, aclTensor* assistInfoForCombineOut,
                           aclTensor* expertTokenNumsOut, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeDispatchTeardown,
                           void* workspace, uint64_t workspaceSize,
                           aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeDispatchV2GetWorkspaceSize,
                           const aclTensor* x, const aclTensor* expertIds,
                           const aclTensor* scalesOptional, const aclTensor* xActiveMaskOptional,
                           const aclTensor* expertScalesOptional,
                           const char* groupEp, int64_t epWorldSize, int64_t epRankId,
                           int64_t moeExpertNum, const char* groupTp, int64_t tpWorldSize,
                           int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
                           int64_t sharedExpertRankNum, int64_t quantMode, int64_t globalBs,
                           int64_t expertTokenNumsType, const char* commAlg,
                           aclTensor* expandXOut, aclTensor* dynamicScalesOut,
                           aclTensor* assistInfoForCombineOut, aclTensor* expertTokenNumsOut,
                           aclTensor* epRecvCountsOut, aclTensor* tpRecvCountsOut, aclTensor* expandScalesOut,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeDispatchV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeDispatchV3GetWorkspaceSize,
                           const aclTensor* x, const aclTensor* expertIds,
                           const aclTensor* scalesOptional, const aclTensor* xActiveMaskOptional,
                           const aclTensor* expertScalesOptional, const aclTensor* elasticInfoOptional,
                           const char* groupEp, int64_t epWorldSize, int64_t epRankId,
                           int64_t moeExpertNum, const char* groupTp, int64_t tpWorldSize,
                           int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
                           int64_t sharedExpertRankNum, int64_t quantMode, int64_t globalBs,
                           int64_t expertTokenNumsType, const char* commAlg,
                           int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum,
                           aclTensor* expandXOut, aclTensor* dynamicScalesOut,
                           aclTensor* assistInfoForCombineOut, aclTensor* expertTokenNumsOut,
                           aclTensor* epRecvCountsOut, aclTensor* tpRecvCountsOut, aclTensor* expandScalesOut,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeDispatchV3,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeDispatchV4GetWorkspaceSize,
                           const aclTensor* x, const aclTensor* expertIds,
                           const aclTensor* scalesOptional, const aclTensor* xActiveMaskOptional,
                           const aclTensor* expertScalesOptional, const aclTensor* elasticInfoOptional,
                           const aclTensor* performanceInfoOptional, const char* groupEp, int64_t epWorldSize, int64_t epRankId,
                           int64_t moeExpertNum, const char* groupTp, int64_t tpWorldSize,
                           int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
                           int64_t sharedExpertRankNum, int64_t quantMode, int64_t globalBs,
                           int64_t expertTokenNumsType, const char* commAlg,
                           int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum,
                           aclTensor* expandXOut, aclTensor* dynamicScalesOut,
                           aclTensor* assistInfoForCombineOut, aclTensor* expertTokenNumsOut,
                           aclTensor* epRecvCountsOut, aclTensor* tpRecvCountsOut, aclTensor* expandScalesOut,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeDistributeDispatchV4,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeFinalizeRoutingV2GetWorkspaceSize,
                           const aclTensor* expandedX, const aclTensor* expandedRowIdx, const aclTensor* x1Optional,
                           const aclTensor* x2Optional, const aclTensor* biasOptional, const aclTensor* scalesOptional,
                           const aclTensor* expertIdxOptional, int64_t dropPadMode, const aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeFinalizeRoutingV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize,
                           const aclTensor* gradY, const aclTensor* expandedRowIdx, const aclTensor* expandedXOptional,
                           const aclTensor* scalesOptional, const aclTensor* expertIdxOptional, const aclTensor* biasOptional,
                           int64_t dropPadMode, int64_t activeNum, int64_t expertNum, int64_t expertCapacity,
                           const aclTensor* gradExpandedXOut, const aclTensor* gradScalesOut, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeFinalizeRoutingV2Grad,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeFusedTopkGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* addNum, const aclTensor* mappingNum, const aclTensor* mappingTable,
                           uint32_t groupNum, uint32_t groupTopk, uint32_t topN, uint32_t topK, uint32_t activateType,
                           bool isNorm, float scale, bool enableExpertMapping, aclTensor* y, aclTensor* indices,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeFusedTopk,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeInitRoutingV2GradGetWorkspaceSize,
                           const aclTensor* gradExpandedX, const aclTensor* expandedRowIdx, int64_t topK, int64_t dropPadMode,
                           int64_t activeNum, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeInitRoutingV2Grad,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeInitRoutingV3GetWorkspaceSize,
                           const aclTensor *x,
                           const aclTensor *expertIdx,
                           const aclTensor *scaleOptional,
                           const aclTensor *offsetOptional,
                           int64_t activeNum,
                           int64_t expertCapacity,
                           int64_t expertNum,
                           int64_t dropPadMode,
                           int64_t expertTokensNumType,
                           bool expertTokensNumFlag,
                           int64_t quantMode,
                           const aclIntArray *activeExpertRangeOptional,
                           int64_t rowIdxType,
                           const aclTensor *expandedXOut,
                           const aclTensor *expandedRowIdxOut,
                           const aclTensor *expertTokensCountOrCumsumOut,
                           const aclTensor *expandedScaleOut,
                           uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeInitRoutingV3,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeTokenPermuteGetWorkspaceSize,
                           const aclTensor* tokens, const aclTensor* indices, int64_t numOutTokens, bool paddedMode,
                           const aclTensor* permuteTokensOut, const aclTensor* sortedIndicesOut, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeTokenPermute,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeTokenPermuteGradGetWorkspaceSize,
                           const aclTensor* permutedOutputGrad, const aclTensor* sortedIndices, int64_t numTopk, bool paddedMode,
                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeTokenPermuteGrad,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize,
                           const aclTensor* tokens, const aclTensor* routingMap, const aclTensor* probsOptional, int64_t numOutTokens,
                           bool dropAndPad, aclTensor* permuteTokensOut, aclTensor* permuteProbsOutOptional, aclTensor* sortedIndicesOut,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeTokenPermuteWithRoutingMap,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeTokenUnpermuteGetWorkspaceSize,
                           const aclTensor* permutedTokens, const aclTensor* sortedIndices, const aclTensor* probsOptional, bool paddedMode,
                           const aclIntArray* restoreShapeOptional, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeTokenUnpermute,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeTokenUnpermuteGradGetWorkspaceSize,
                           const aclTensor* permuteTokens, const aclTensor* unpermutedTokensGrad, const aclTensor* sortedIndices,
                           const aclTensor* probsOptional, bool paddedMode, const aclIntArray* restoreShapeOptional,
                           aclTensor* permutedTokensGradOut, aclTensor* probsGradOut, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeTokenUnpermuteGrad,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize,
                           const aclTensor* permutedTokens,
                           const aclTensor* sortedIndices,
                           const aclTensor* routingMapOptional,
                           const aclTensor* probsOptional,
                           bool paddedMode,
                           const aclIntArray* restoreShapeOptional,
                           aclTensor* unpermutedTokens, aclTensor* outIndex, aclTensor* permuteTokenId, aclTensor* permuteProbs,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeTokenUnpermuteWithRoutingMap,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeUpdateExpertGetWorkspaceSize,
                           const aclTensor* expertIds, const aclTensor* eplbTable, const aclTensor* expertScalesOptional,
                           const aclTensor* pruningThresholdOptional, const aclTensor* activeMaskOptional, int64_t localRankId,
                           int64_t worldSize, int64_t balanceMode, aclTensor* balancedExpertIds, aclTensor* balancedActiveMask,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMoeUpdateExpert,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMseLossGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* target, int64_t reduction,
                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMseLoss,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMseLossBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self,
                           const aclTensor* target, int64_t reduction, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMseLossBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMseLossOutGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* target, int64_t reduction,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMseLossOut,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMulGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* other, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMul,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMulsGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* other, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMuls,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize,
                           const aclTensor* value, const aclTensor* spatialShape, const aclTensor* levelStartIndex, const aclTensor* location,
                           const aclTensor* attnWeight, const aclTensor* gradOutput, aclTensor* gradValue, aclTensor* gradLocation,
                           aclTensor* gradAttnWeight, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMultiScaleDeformableAttentionGrad,
                           void* workspace, uint64_t workspaceSize,
                           aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMultiScaleDeformableAttnFunctionGetWorkspaceSize,
                           const aclTensor* value, const aclTensor* spatialShape, const aclTensor* levelStartIndex, const aclTensor* location,
                           const aclTensor* attnWeight, aclTensor* output, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMultiScaleDeformableAttnFunction,
                           void* workspace, uint64_t workspaceSize,
                           aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMultilabelMarginLossGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* target,
                           int64_t reduction, aclTensor* out, aclTensor* isTarget,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMultilabelMarginLoss,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMultinomialGetWorkspaceSize,
                           const aclTensor* self, int64_t numsamples, bool replacement,
                           int64_t seed, int64_t offset, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMultinomial,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMultinomialTensorGetWorkspaceSize,
                           const aclTensor* self, int64_t numsamples, bool replacement,
                           const aclTensor* seedTensor, const aclTensor* offsetTensor, int64_t offset,
                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMultinomialTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnMvGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* vec, aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnMv,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnNLLLossGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* target, const aclTensor* weight, int64_t reduction, int64_t ignoreIndex,
                           aclTensor* out, aclTensor* totalWeightOut, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnNLLLoss,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnNLLLoss2dGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* target, const aclTensor* weight, int64_t reduction, int64_t ignoreIndex,
                           aclTensor* out, aclTensor* totalWeightOut, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnNLLLoss2d,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnNLLLoss2dBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, const aclTensor* weight,
                           int64_t reduction, int64_t ignoreIndex, aclTensor* totalWeight, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnNLLLoss2dBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnNLLLossBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, const aclTensor* weight,
                           int64_t reduction, int64_t ignoreIndex, const aclTensor* totalWeight, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnNLLLossBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnNanMedianGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnNanMedian,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnNanMedianDimGetWorkspaceSize,
                           const aclTensor* self, int64_t dim, bool keepDim,
                           aclTensor* valuesOut, aclTensor* indicesOut,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnNanMedianDim,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnNanToNumGetWorkspaceSize,
                           const aclTensor* self, float nan, float posinf, float neginf,
                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnNanToNum,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnNegGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnNeg,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnNonMaxSuppressionGetWorkspaceSize,
                           const aclTensor* boxes, const aclTensor* scores,
                           aclIntArray* maxOutputBoxesPerClass,
                           aclFloatArray* iouThreshold, aclFloatArray* scoreThreshold,
                           int32_t centerPointBox, aclTensor* selectedIndices,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnNonMaxSuppression,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnNonzeroGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnNonzero,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnNonzeroV2GetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnNonzeroV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnNormGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* pScalar, const aclIntArray* dim, bool keepdim, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnNorm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnNormRopeConcatGetWorkspaceSize,
                           const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *encoderQuery,
                           const aclTensor *encoderKey, const aclTensor *encoderValue, const aclTensor *normQueryWeight,
                           const aclTensor *normQueryBias, const aclTensor *normKeyWeight, const aclTensor *normKeyBias,
                           const aclTensor *normAddedQueryWeight, const aclTensor *normAddedQueryBias, const aclTensor *normAddedKeyWeight,
                           const aclTensor *normAddedKeyBias, const aclTensor *ropeSin, const aclTensor *ropeCos, int64_t normType,
                           int64_t normAddedType, int64_t ropeType, int64_t concatOrder, double eps, bool isTraining,
                           const aclTensor *queryOutput, const aclTensor *keyOutput, const aclTensor *valueOutput,
                           const aclTensor *normQueryMean, const aclTensor *normQueryRstd, const aclTensor *normKeyMean,
                           const aclTensor *normKeyRstd, const aclTensor *normAddedQueryMean, const aclTensor *normAddedQueryRstd,
                           const aclTensor *normAddedKeyMean, const aclTensor *normAddedKeyRstd, uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnNormRopeConcat,
                           void *workspace, uint64_t workspaceSize,
                           aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnNormalFloatFloatGetWorkspaceSize,
                           float mean, float std, int64_t seed, int64_t offset, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnNormalFloatFloat,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnNormalFloatTensorGetWorkspaceSize,
                           float mean, const aclTensor* std, int64_t seed, int64_t offset, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnNormalFloatTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnNormalTensorFloatGetWorkspaceSize,
                           const aclTensor* mean, float std, int64_t seed, int64_t offset, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnNormalTensorFloat,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnNormalTensorTensorGetWorkspaceSize,
                           const aclTensor* mean, const aclTensor* std, int64_t seed, int64_t offset, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnNormalTensorTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnNpuFormatCastGetWorkspaceSize,
                           const aclTensor* srcTensor, aclTensor* dstTensor, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnNpuFormatCast,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnNpuFormatCastCalculateSizeAndFormat,
                           const aclTensor* srcTensor, const int dstFormat, int additionalDtype, int64_t** dstShape,
                           uint64_t* dstShapeSize, int* actualFormat)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnPdistGetWorkspaceSize,
                           const aclTensor* self, float p, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnPdist,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnPdistForwardGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* pScalar, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnPdistForward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnPermuteGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* dims, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnPermute,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnPolarGetWorkspaceSize,
                           const aclTensor* input, const aclTensor* angle, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnPolar,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnPowScalarTensorGetWorkspaceSize,
                           const aclScalar* self, const aclTensor* exponent,
                           const aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnPowScalarTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnPowTensorScalarGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* exponent,
                           const aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnPowTensorScalar,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnPowTensorTensorGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* exponent,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnPowTensorTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnPrecisionCompareGetWorkspaceSize,
                           const aclTensor *golden, const aclTensor *realdata,
                           aclTensor *out, uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnPrecisionCompare,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnPrecisionCompareV2GetWorkspaceSize,
                           const aclTensor* golden, const aclTensor* realdata,
                           uint32_t detect_type, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnPrecisionCompareV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnPreluGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* weight, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnPrelu,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnPreluBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self,
                           const aclTensor* weight, aclTensor* gradInput,
                           aclTensor* gradWeight, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnPreluBackward,
                           void* workspace, uint64_t workspace_size, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnQkvRmsNormRopeCacheGetWorkspaceSize,
                           const aclTensor* qkv, const aclTensor* qGamma, const aclTensor* kGamma, const aclTensor* cos, const aclTensor* sin,
                           const aclTensor* index, aclTensor* qOut, aclTensor* kCache, aclTensor* vCache, const aclTensor* kScaleOptional, const aclTensor* vScaleOptional,
                           const aclTensor* kOffsetOptional, const aclTensor* vOffsetOptional, const aclIntArray* qkvSize, const aclIntArray* headNums, double epsilon, char* cacheModeOptional,
                           const aclTensor* qOutBeforeQuant, const aclTensor* kOutBeforeQuant, const aclTensor* vOutBeforeQuant, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnQkvRmsNormRopeCache,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnQrGetWorkspaceSize,
                           const aclTensor* self, bool some, aclTensor* Q, aclTensor* R,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnQr,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantAllReduceGetWorkspaceSize,
                           const aclTensor* x,
                           const aclTensor* scales,
                           const char* group,
                           const char* reduceOp,
                           aclTensor* output,
                           uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantAllReduce,
                           void* workspace,
                           uint64_t workspaceSize,
                           aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2,
                           const aclTensor* x1ScaleOptional, const aclTensor* x2Scale,
                           aclTensor* yRef, bool transposeX1,
                           bool transposeX2, int64_t groupSize,
                           uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantBatchMatmulInplaceAdd,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantConvolutionGetWorkspaceSize,
                           const aclTensor* input, const aclTensor* weight,
                           const aclTensor* bias, const aclTensor *scale,
                           const aclTensor *offset, const aclIntArray* stride,
                           const aclIntArray* padding, const aclIntArray* dilation,
                           bool transposed, const aclIntArray* outputPadding,
                           int64_t groups, int32_t offsetx,
                           const char* roundMode, aclTensor* output,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantConvolution,
                           void *workspace, const uint64_t workspaceSize, aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantConvolutionWeightNzGetWorkspaceSize,
                           const aclTensor* input, const aclTensor* weight,
                           const aclTensor* bias, const aclTensor *scale,
                           const aclTensor *offset, const aclIntArray* stride,
                           const aclIntArray* padding, const aclIntArray* dilation,
                           bool transposed, const aclIntArray* outputPadding,
                           int64_t groups, int32_t offsetx,
                           const char* roundMode, aclTensor* output,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantConvolutionWeightNz,
                           void *workspace, const uint64_t workspaceSize, aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantGroupedMatMulAlltoAllvGetWorkspaceSize,
                           const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale, const aclTensor *gmmWeightScale,
                           const aclTensor *gmmXOffsetOptional, const aclTensor *gmmWeightOffsetOptional,
                           const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
                           const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
                           const aclTensor *mmXOffsetOptional, const aclTensor *mmWeightOffsetOptional,
                           const aclTensor *commQuantScaleOptional, int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, int64_t mmXQuantMode,
                           int64_t mmWeightQuantMode, int64_t commQuantMode, int64_t commQuantDtypeOptional,
                           int64_t groupSize, const char *group, int64_t epWorldSize, const aclIntArray *sendCounts,
                           const aclIntArray *recvCounts, bool transGmmWeight, bool transMmWeight, const aclTensor *y,
                           const aclTensor *mmYOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantGroupedMatMulAlltoAllv,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantGroupedMatmulDequantGetWorkspaceSize,
                           const aclTensor *x, const aclTensor *weight,
                           const aclTensor *weightScale, const aclTensor *groupList, const aclTensor *biasOptional,
                           const aclTensor *xScaleOptional, const aclTensor *xOffsetOptional,
                           const aclTensor *smoothScaleOptional,
                           char *xQuantMode, bool transposeWeight, const aclTensor *out,
                           uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantGroupedMatmulDequant,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantMatmulGetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, float deqScale, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantMatmul,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantMatmulAllReduceGetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x3, const aclTensor* dequantScale,
                           const char* group, const char* reduceOp, int64_t commTurn, int64_t streamMode, const aclTensor* output,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantMatmulAllReduce,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantMatmulAllReduceAddRmsNormGetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* dequantScale,
                           const aclTensor* residual, const aclTensor* gamma, double epsilon, const char* group, const char* reduceOp,
                           int64_t commTurn, int64_t streamMode, const aclTensor* y, const aclTensor* normOut, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantMatmulAllReduceAddRmsNorm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantMatmulAllReduceV2GetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional, const aclTensor* x3Optional,
                           const aclTensor* dequantScale, const aclTensor* pertokenScaleOptional, const char* group, const char* reduceOp,
                           int64_t commTurn, int64_t streamMode, const aclTensor* output, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantMatmulAllReduceV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantMatmulAllReduceV3GetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional, const aclTensor* x3Optional,
                           const aclTensor* dequantScale, const aclTensor* pertokenScaleOptional, const aclTensor* commQuantScale1Optional,
                           const aclTensor* commQuantScale2Optional, const char* group, const char* reduceOp, int64_t commTurn,
                           int64_t streamMode, const aclTensor* output, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantMatmulAllReduceV3,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantMatmulAllReduceV4GetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional, const aclTensor* x3Optional,
                           const aclTensor* x1ScaleOptional, const aclTensor* x2Scale, const aclTensor* commQuantScale1Optional,
                           const aclTensor* commQuantScale2Optional, const char* group, const char* reduceOp, int64_t commTurn,
                           int64_t streamMode, int64_t groupSize, int64_t commQuantMode, const aclTensor* output, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantMatmulAllReduceV4,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantMatmulDequantGetWorkspaceSize,
                           const aclTensor *x, const aclTensor *weight,
                           const aclTensor *weightScale, const aclTensor *biasOptional,
                           const aclTensor *xScaleOptional, const aclTensor *xOffsetOptional,
                           const aclTensor *smoothScaleOptional,
                           char *xQuantMode, bool transposeWeight, const aclTensor *out,
                           uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantMatmulDequant,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale, const aclTensor* x2Scale,
                           const aclTensor* yScale, const aclTensor* x1Offset, const aclTensor* x2Offset, const aclTensor* yOffset,
                           const aclTensor* bias, bool transposeX1, bool transposeX2,
                           int64_t groupSize, const aclIntArray* dims, bool keepDims,
                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantMatmulReduceSumWeightNz,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantMatmulV2GetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* deqScale, bool adjX1, bool adjX2,
                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantMatmulV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantMatmulV3GetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2,
                           const aclTensor* scale, const aclTensor* offset,
                           const aclTensor* bias, bool transposeX1, bool transposeX2,
                           const aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantMatmulV3,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantMatmulV4GetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2,
                           const aclTensor* scale, const aclTensor* offset,
                           const aclTensor* pertokenScaleOptional, const aclTensor* bias,
                           bool transposeX1, bool transposeX2, const aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantMatmulV4,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantMatmulV5GetWorkspaceSize,
                           const aclTensor *x1, const aclTensor *x2,
                           const aclTensor *x1Scale, const aclTensor *x2Scale,
                           const aclTensor *yScale, const aclTensor *x1Offset,
                           const aclTensor *x2Offset, const aclTensor *yOffset,
                           const aclTensor *bias, bool transposeX1, bool transposeX2,
                           int64_t groupSize, aclTensor *out, uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantMatmulV5,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantMatmulWeightNzGetWorkspaceSize,
                           const aclTensor *x1, const aclTensor *x2,
                           const aclTensor *x1Scale, const aclTensor *x2Scale,
                           const aclTensor *yScale, const aclTensor *x1Offset,
                           const aclTensor *x2Offset, const aclTensor *yOffset,
                           const aclTensor *bias, bool transposeX1,
                           bool transposeX2, int64_t groupSize, aclTensor *out,
                           uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantMatmulWeightNz,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantReduceScatterGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* scales,
                           const char* group, const char* reduceOp,
                           aclTensor* output, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantReduceScatter,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantizeGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* scales,
                           const aclTensor* zeroPoints, aclDataType dtype, int32_t axis, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantize,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantizedBatchNormGetWorkspaceSize,
                           const aclTensor* input, const aclTensor* mean, const aclTensor* var, const aclScalar* inputScale,
                           const aclScalar* inputZeroPoint, const aclScalar* outputScale, const aclScalar* outputZeroPoint, aclTensor* weight,
                           aclTensor* bias, float epsilon, aclTensor* output, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnQuantizedBatchNorm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRReluWithNoiseGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* noise,
                           const aclScalar* lower, const aclScalar* upper, bool training,
                           int64_t seed, int64_t offset, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRReluWithNoise,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRangeGetWorkspaceSize,
                           const aclScalar* start, const aclScalar* end, const aclScalar* step, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRange,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRealGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnReal,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnReciprocalGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnReciprocal,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRecurrentGatedDeltaRuleGetWorkspaceSize,
                           const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *beta, aclTensor *stateRef,
                           const aclTensor *actualSeqLengths, const aclTensor *ssmStateIndices, const aclTensor *g, const aclTensor *gk,
                           const aclTensor *numAcceptedTokens, float scaleValue, aclTensor *out, uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRecurrentGatedDeltaRule,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnReflectionPad1dGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* padding,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnReflectionPad1d,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnReflectionPad1dBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* padding, aclTensor* gradInput,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnReflectionPad1dBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnReflectionPad2dGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* padding,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnReflectionPad2d,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnReflectionPad2dBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self,
                           const aclIntArray* padding, aclTensor* gradInput,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnReflectionPad2dBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnReflectionPad3dGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* padding,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnReflectionPad3d,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnReflectionPad3dBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self,
                           const aclIntArray* padding, aclTensor* gradInput,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnReflectionPad3dBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnReluGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRelu,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRemainderScalarTensorGetWorkspaceSize,
                           const aclScalar* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRemainderScalarTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRemainderTensorScalarGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRemainderTensorScalar,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRemainderTensorTensorGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRemainderTensorTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRenormGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* p, int64_t dim, const aclScalar* maxNorm, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRenorm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRepeatGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* repeats, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRepeat,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRepeatInterleaveGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* repeats, int64_t outputSize, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRepeatInterleave,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRepeatInterleaveIntGetWorkspaceSize,
                           const aclTensor* self, int64_t repeats, int64_t outputSize, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRepeatInterleaveInt,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRepeatInterleaveIntWithDimGetWorkspaceSize,
                           const aclTensor* self, int64_t repeats, int64_t dim, int64_t outputSize, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRepeatInterleaveIntWithDim,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRepeatInterleaveTensorGetWorkspaceSize,
                           const aclTensor* repeats, int64_t outputSize, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRepeatInterleaveTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRepeatInterleaveWithDimGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* repeats, int64_t dim, int64_t outputSize, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRepeatInterleaveWithDim,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnReplicationPad1dGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* padding,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnReplicationPad1d,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnReplicationPad1dBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self,
                           const aclIntArray* padding, aclTensor* gradInput,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnReplicationPad1dBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnReplicationPad2dGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* padding,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnReplicationPad2d,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnReplicationPad2dBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self,
                           const aclIntArray* padding, aclTensor* gradInput,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnReplicationPad2dBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnReplicationPad3dGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* padding,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnReplicationPad3d,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnReplicationPad3dBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self,
                           const aclIntArray* padding, aclTensor* gradInput,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnReplicationPad3dBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnResizeGetWorkspaceSize,
                           const aclTensor* self, const aclFloatArray* scales, const char* mode, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnResize,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRightShiftGetWorkspaceSize,
                           const aclTensor *input, const aclTensor *shiftBits,
                           aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRightShift,
                           void *workspace, uint64_t workspaceSize,
                           aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRmsNormQuantGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* gamma, const aclTensor* beta, const aclTensor* scale, const aclTensor* offset,
                           double epsilon, aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRmsNormQuant,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRoiAlignGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* rois,
                           const aclTensor* batchIndices, const char* mode, int outputHeight,
                           int outputWidth, int samplingRatio, float spatialScale,
                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRoiAlign,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRoiAlignV2GetWorkspaceSize,
                           const aclTensor* self, const aclTensor* boxes, int64_t pooledHeight,
                           int64_t pooledWidth, float spatialScale, int64_t samplingRatio, bool aligned, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRoiAlignV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRoiAlignV2BackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* boxes,
                           const aclIntArray* inputShape, int64_t pooledHeight, int64_t pooledWidth,
                           float spatialScale, int64_t samplingRatio, bool aligned, aclTensor* gradInput,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRoiAlignV2Backward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRoiPoolingGradWithArgMaxGetWorkspaceSize,
                           const aclTensor *gradOutput,
                           const aclTensor *gradInputRef,
                           const aclTensor *rois,
                           const aclTensor *argmax,
                           int64_t pooledH,
                           int64_t pooledW,
                           double spatialScale,
                           uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRoiPoolingGradWithArgMax,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRoiPoolingWithArgMaxGetWorkspaceSize,
                           const aclTensor *x, const aclTensor *rois,
                           int64_t pooled_h, int64_t pooled_w, float spatial_scale_h, float spatial_scale_w, aclTensor *y, aclTensor *argmax,
                           uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRoiPoolingWithArgMax,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRollGetWorkspaceSize,
                           const aclTensor* x, const aclIntArray* shifts, const aclIntArray* dims, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRoll,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRopeWithSinCosCacheGetWorkspaceSize,
                           const aclTensor* positions, const aclTensor* queryIn, const aclTensor* keyIn, const aclTensor* cosSinCache,
                           const aclIntArray* mropeSection, int64_t headSize, bool isNeoxStyle, aclTensor* queryOut, aclTensor* keyOut,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRopeWithSinCosCache,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRopeWithSinCosCacheV2GetWorkspaceSize,
                           const aclTensor* positions, const aclTensor* queryIn, const aclTensor* keyIn, const aclTensor* cosSinCache,
                           const aclIntArray* mropeSection, int64_t headSize, bool isNeoxStyle, int64_t cacheMode, aclTensor* queryOut, aclTensor* keyOut,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRopeWithSinCosCacheV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRotaryPositionEmbeddingGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* cos,
                           const aclTensor* sin, int64_t mode, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRotaryPositionEmbedding,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRotaryPositionEmbeddingV2GetWorkspaceSize,
                           const aclTensor* x, const aclTensor* cos,
                           const aclTensor* sin, int64_t mode, const aclTensor* rotate,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRotaryPositionEmbeddingV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRoundGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRound,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRoundDecimalsGetWorkspaceSize,
                           const aclTensor* self, int64_t decimals, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRoundDecimals,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRsqrtGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRsqrt,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRsubGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* other, const aclScalar* alpha, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRsub,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRsubsGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* other, const aclScalar* alpha, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRsubs,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSWhereGetWorkspaceSize,
                           const aclTensor* condition, const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSWhere,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnScaleGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* scale, const aclTensor* bias,
                           int64_t axis, int64_t numAxes, bool scaleFromBlob,
                           aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnScale,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnScaledMaskedSoftmaxGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* mask, double scale, bool fixedTriuMask, aclTensor* y, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnScaledMaskedSoftmax,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnScaledMaskedSoftmaxBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* y, const aclTensor* mask, double scale, bool fixTriuMask,
                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnScaledMaskedSoftmaxBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnScatterGetWorkspaceSize,
                           const aclTensor* self, int64_t dim, const aclTensor* index, const aclTensor* src, int64_t reduce, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnScatter,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnScatterAddGetWorkspaceSize,
                           const aclTensor* self, int64_t dim, const aclTensor* index,
                           const aclTensor* src, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnScatterAdd,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnScatterNdGetWorkspaceSize,
                           const aclTensor* data, const aclTensor* indices,
                           const aclTensor* updates, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnScatterNd,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnScatterNdUpdateGetWorkspaceSize,
                           aclTensor* varRef, const aclTensor* indices,
                           const aclTensor* updates, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnScatterNdUpdate,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnScatterValueGetWorkspaceSize,
                           const aclTensor* self, int64_t dim, const aclTensor* index, const aclScalar* value, int64_t reduce, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnScatterValue,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSearchSortedGetWorkspaceSize,
                           const aclTensor* sortedSequence, const aclTensor* self,
                           const bool outInt32, const bool right, const aclTensor* sorter,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSearchSorted,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSearchSortedsGetWorkspaceSize,
                           const aclTensor* sortedSequence, const aclScalar* self,
                           const bool outInt32, const bool right, const aclTensor* sorter,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSearchSorteds,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSeluGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSelu,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSeluBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* result,
                           aclTensor* gradInput, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSeluBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnShrinkGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* lambd, const aclScalar* bias, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnShrink,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSigmoidGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSigmoid,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSigmoidBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* output,
                           aclTensor* gradInput, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSigmoidBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSignGetWorkspaceSize,
                           const aclTensor* self, aclTensor* result, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSign,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSignBitsPackGetWorkspaceSize,
                           const aclTensor* self, int64_t size,
                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSignBitsPack,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSignBitsUnpackGetWorkspaceSize,
                           const aclTensor* self, int64_t size, aclDataType dtype, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSignBitsUnpack,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSignbitGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSignbit,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSilentCheckGetWorkspaceSize,
                           const aclTensor *val, aclTensor *inputGradRef,
                           aclTensor *sfdaRef, aclTensor *stepRef, const int32_t cMinSteps,
                           const float cThreshL1, const float cCoeffL1,
                           const float cThreshL2, const float cCoeffL2,
                           const int32_t npuAsdDetect, aclTensor* result,
                           uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSilentCheck,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSilentCheckV2GetWorkspaceSize,
                           const aclTensor *val, const aclTensor *max, aclTensor *avgRef, aclTensor *inputGradRef, aclTensor *stepRef,
                           aclIntArray *dstSize, aclIntArray *dstStride, aclIntArray *dstOffset, float cThreshL1,
                           float cThreshL2, float beta1, int32_t npuAsdDetect, aclTensor* result,
                           uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSilentCheckV2,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSiluGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSilu,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSiluBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self,
                           aclTensor* gradInput, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSiluBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSinGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSin,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSincGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSinc,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSinhGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSinh,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSinkhornGetWorkspaceSize,
                           const aclTensor* cost, const aclScalar* tol, aclTensor* p, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSinkhorn,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSliceGetWorkspaceSize,
                           const aclTensor* self, int64_t dim, int64_t start, int64_t end, int64_t step, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSlice,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSliceV2GetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* starts, const aclIntArray* ends, const aclIntArray* axes,
                           const aclIntArray* steps, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSliceV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSlogdetGetWorkspaceSize,
                           const aclTensor* self, aclTensor* signOut, aclTensor* logOut,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSlogdet,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSmoothL1LossGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* target,
                           int64_t reduction, float beta, aclTensor* result,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSmoothL1Loss,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSmoothL1LossBackwardGetWorkspaceSize,
                           const aclTensor* gradOut, const aclTensor* self,
                           const aclTensor* target, int64_t reduction, float beta,
                           aclTensor* gradInput, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSmoothL1LossBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSoftMarginLossGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* target,
                           int64_t reduction, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSoftMarginLoss,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSoftMarginLossBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self,
                           const aclTensor* target, int64_t reduction,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSoftMarginLossBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSoftmaxGetWorkspaceSize,
                           const aclTensor* self, int64_t dim, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSoftmax,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSoftmaxBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* output,
                           int64_t dim, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSoftmaxBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSoftmaxCrossEntropyWithLogitsGetWorkspaceSize,
                           const aclTensor* features, aclTensor* labels, aclTensor* loss, aclTensor* backprop, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSoftmaxCrossEntropyWithLogits,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSoftplusGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* beta,
                           const aclScalar* threshold, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSoftplus,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSoftplusBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self,
                           const aclScalar* beta, const aclScalar* threshold,
                           aclTensor* gradInput, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSoftplusBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSoftshrinkGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* lambd, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSoftshrink,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSoftshrinkBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self,
                           const aclScalar* lambda, aclTensor* gradInput,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSoftshrinkBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSparse4to2QuantMatmulWeightNzGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* sparseWeight, const aclTensor* index, const aclTensor* xScale,
                           const aclTensor* sparseWeightScale, const aclTensor* biasOptional, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSparse4to2QuantMatmulWeightNz,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSplitTensorGetWorkspaceSize,
                           const aclTensor* self, uint64_t splitSections, int64_t dim,
                           aclTensorList* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSplitTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSplitWithSizeGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* splitSize,
                           int64_t dim, aclTensorList* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSplitWithSize,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSqrtGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** opExecutor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSqrt,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* opExecutor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSquareGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSquare,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnStridedSliceGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* begin, const aclIntArray* end, const aclIntArray* strides,
                           int64_t beginMask, int64_t endMask, int64_t ellipsisMask, int64_t newAxisMask, int64_t shrinkAxisMask,
                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnStridedSlice,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSubGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* other, const aclScalar* alpha, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSub,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSubsGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* other, const aclScalar* alpha, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSubs,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSumGetWorkspaceSize,
                           const aclTensorList* tensors, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSum,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSvdGetWorkspaceSize,
                           const aclTensor *input, const bool fullMatrices, const bool computeUV,
                           aclTensor *sigma, aclTensor *u, aclTensor *v, uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSvd,
                           void *workspace, uint64_t workspaceSize,
                           aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSwishGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* betaOptional, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSwish,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSwishBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* self,
                           const aclScalar* betaOptional, aclTensor* gradInput,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSwishBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnSyncBatchNormGatherStatsGetWorkspaceSize,
                           const aclTensor* totalSum, const aclTensor* totalSquareSum, const aclTensor* sampleCount, aclTensor* mean,
                           aclTensor* variance, float momentum, float eps, aclTensor* batchMean, aclTensor* batchInvstd,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnSyncBatchNormGatherStats,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnTakeGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* index, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnTake,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnTanGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnTan,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnTanhGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnTanh,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnTanhBackwardGetWorkspaceSize,
                           const aclTensor* gradOutput, const aclTensor* output,
                           aclTensor* gradInput, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnTanhBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnTfScatterAddGetWorkspaceSize,
                           aclTensor* varRef, const aclTensor* indices, const aclTensor* updates, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnTfScatterAdd,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnThnnFusedLstmCellGetWorkspaceSize,
                           const aclTensor    *inputGates,
                           const aclTensor    *hiddenGates,
                           const aclTensor    *cx,
                           const aclTensor    *inputBiasOptional,
                           const aclTensor    *hiddenBiasOptional,
                           aclTensor          *hyOut,
                           aclTensor          *cyOut,
                           aclTensor          *storageOut,
                           uint64_t           *workspaceSize,
                           aclOpExecutor      **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnThnnFusedLstmCell,
                           void               *workspace,
                           uint64_t            workspaceSize,
                           aclOpExecutor      *executor,
                           const aclrtStream   stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnThnnFusedLstmCellBackwardGetWorkspaceSize,
                           const aclTensor *gradHyOptional,
                           const aclTensor *gradCOptional,
                           const aclTensor *cx,
                           const aclTensor *cy,
                           const aclTensor *storage,
                           bool hasBias,
                           aclTensor *gradGatesOut,
                           aclTensor *gradCxOut,
                           aclTensor *gradBiasOut,
                           uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnThnnFusedLstmCellBackward,
                           void *workspace,
                           uint64_t workspaceSize,
                           aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnThreeInterpolateBackwardGetWorkspaceSize,
                           const aclTensor* grad_x, const aclTensor* idx, const aclTensor* weight, int m, aclTensor* grad_y,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnThreeInterpolateBackward,
                           void* workspace, uint64_t workspace_size, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnThresholdGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* threshold, const aclScalar* value, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnThreshold,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnThresholdBackwardGetWorkspaceSize,
                           const aclTensor *gradOutput, const aclTensor *self,
                           const aclScalar *threshold, aclTensor *out,
                           uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnThresholdBackward,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnTopkGetWorkspaceSize,
                           const aclTensor* self, int64_t k, int64_t dim, bool largest,
                           bool sorted, aclTensor* valuesOut, aclTensor* indicesOut,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnTopk,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnTraceGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnTrace,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnTransConvolutionWeightGetWorkspaceSize,
                           const aclTensor* weightIn, bool transposed,
                           const int64_t groups, aclTensor* weightOut,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnTransConvolutionWeight,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnTransMatmulWeightGetWorkspaceSize,
                           aclTensor* mmWeightRef, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnTransMatmulWeight,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnTransQuantParam,
                           const float* scaleArray, uint64_t scaleSize, const float* offsetArray,
                           uint64_t offsetSize, uint64_t** quantParam, uint64_t* quantParamSize)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnTransQuantParamV2GetWorkspaceSize,
                           const aclTensor* scale, const aclTensor* offset, const aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnTransQuantParamV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnTransQuantParamV3GetWorkspaceSize,
                           const aclTensor* scale, const aclTensor* offset, int64_t roundMode, const aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnTransQuantParamV3,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnTransSparse4to2Para,
                           const int8_t* weight, aclIntArray* shape, int8_t** sparseWeight, int64_t** sparseWeightDims,
                           uint64_t* sparseWeightDimsNum, uint8_t** index, int64_t** indexDims, uint64_t* indexDimsNum)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnTransposeBatchMatMulGetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2,
                           const aclTensor* bias, const aclTensor* scale,
                           const aclIntArray* permX1, const aclIntArray* permX2,
                           const aclIntArray* permY, int8_t cubeMathType,
                           const int32_t batchSplitFactor, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnTransposeBatchMatMul,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnTransposeQuantBatchMatMulGetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x1Scale, const aclTensor* x2Scale,
                           const int32_t dtype, const int32_t groupSize, const aclIntArray* permX1, const aclIntArray* permX2,
                           const aclIntArray* permY, const int32_t batchSplitFactor, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnTransposeQuantBatchMatMul,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnTriangularSolveGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* A, bool upper,
                           bool transpose, bool unitriangular, aclTensor* xOut,
                           aclTensor* mOut, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnTriangularSolve,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnTrilGetWorkspaceSize,
                           const aclTensor* self, int64_t diagonal, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnTril,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnTriuGetWorkspaceSize,
                           const aclTensor* self, int64_t diagonal, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnTriu,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnTruncGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnTrunc,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUnfoldGradGetWorkspaceSize,
                           const aclTensor* gradOut, const aclIntArray* inputSizes, int64_t dim, int64_t size, int64_t step,
                           const aclTensor* gradIn, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUnfoldGrad,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUniqueGetWorkspaceSize,
                           const aclTensor* self, bool sorted, bool returnInverse,
                           aclTensor* valueOut, aclTensor* inverseOut, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUnique,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUnique2GetWorkspaceSize,
                           const aclTensor* self, bool sorted, bool returnInverse,
                           bool returnCounts, aclTensor* valueOut, aclTensor* inverseOut,
                           aclTensor* countsOut, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUnique2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUniqueConsecutiveGetWorkspaceSize,
                           const aclTensor* self, bool returnInverse,
                           bool returnCounts, int64_t dim, aclTensor* valueOut,
                           aclTensor* inverseOut, aclTensor* countsOut,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUniqueConsecutive,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUniqueDimGetWorkspaceSize,
                           const aclTensor* self, bool sorted, bool returnInverse,
                           int64_t dim, aclTensor* valueOut, aclTensor* inverseOut,
                           aclTensor* countsOut, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUniqueDim,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleBicubic2dGetWorkspaceSize,
                           const aclTensor *self, const aclIntArray *outputSize,
                           const bool alignCorners, const double scalesH, const double scalesW, aclTensor *out, uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleBicubic2d,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleBicubic2dAAGetWorkspaceSize,
                           const aclTensor *x, const aclIntArray *outputSize,
                           const bool alignCorners, const double scalesH, const double scalesW, aclTensor *out, uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleBicubic2dAA,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleBicubic2dBackwardGetWorkspaceSize,
                           const aclTensor *gradOut,
                           const aclIntArray *outputSize, const aclIntArray *inputSize, const bool alignCorners, double scalesH,
                           double scalesW, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleBicubic2dBackward,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleBilinear2dGetWorkspaceSize,
                           const aclTensor *self, const aclIntArray *outputSize,
                           const bool alignCorners, const double scalesH, const double scalesW, aclTensor *out, uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleBilinear2d,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleBilinear2dAAGetWorkspaceSize,
                           const aclTensor *input, const aclIntArray *outputSize,
                           bool alignCorners, double scalesH, double scalesW, aclTensor *out, uint64_t *workspaceSize,
                           aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleBilinear2dAA,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleBilinear2dBackwardGetWorkspaceSize,
                           const aclTensor *gradOut,
                           const aclIntArray *outputSize, const aclIntArray *inputSize, bool alignCorners, double scalesH, double scalesW,
                           aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleBilinear2dBackward,
                           void *workspace, uint64_t workspace_size, aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleBilinear2dBackwardV2GetWorkspaceSize,
                           const aclTensor *gradOut,
                           const aclIntArray *outputSize, const aclIntArray *inputSize, bool alignCorners, double scalesH, double scalesW,
                           aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleBilinear2dBackwardV2,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleLinear1dGetWorkspaceSize,
                           const aclTensor *self, const aclIntArray *outputSize,
                           const bool alignCorners, const double scale, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleLinear1d,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleLinear1dBackwardGetWorkspaceSize,
                           const aclTensor *gradOut,
                           const aclIntArray *outputSize, const aclIntArray *inputSize, bool alignCorners, double scales, aclTensor *out,
                           uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleLinear1dBackward,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleNearest1dGetWorkspaceSize,
                           const aclTensor *self, const aclIntArray *outputSize,
                           aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleNearest1d,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleNearest1dBackwardGetWorkspaceSize,
                           const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize, double scales,
                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleNearest1dBackward,
                           void* workspace, uint64_t workspace_size, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleNearest1dV2GetWorkspaceSize,
                           const aclTensor *self, const aclIntArray *outputSize,
                           float scaleL, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleNearest1dV2,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleNearest2dGetWorkspaceSize,
                           const aclTensor *self, const aclIntArray *outputSize,
                           aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleNearest2d,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleNearest2dBackwardGetWorkspaceSize,
                           const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize, double scalesH,
                           double scalesW, aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleNearest2dBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleNearest2dV2GetWorkspaceSize,
                           const aclTensor *self, const aclIntArray *outputSize,
                           float scalesH, float scalesW, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleNearest2dV2,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleNearest3dGetWorkspaceSize,
                           const aclTensor *self, const aclIntArray *outputSize,
                           double scalesD, double scalesH, double scalesW, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleNearest3d,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleNearest3dBackwardGetWorkspaceSize,
                           const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize, double scalesD,
                           double scalesH, double scalesW, aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleNearest3dBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleNearestExact1dGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* outputSize, double scales, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleNearestExact1d,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleNearestExact2dGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* outputSize, double scalesH, double scalesW, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleNearestExact2d,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleNearestExact3dGetWorkspaceSize,
                           const aclTensor *self, const aclIntArray *outputSize,
                           double scalesD, double scalesH, double scalesW, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleNearestExact3d,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleNearestExact3dBackwardGetWorkspaceSize,
                           const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize, double scalesD,
                           double scalesH, double scalesW, aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleNearestExact3dBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleTrilinear3dGetWorkspaceSize,
                           const aclTensor* self, const aclIntArray* outputSize, bool alignCorners, double scalesD, double scalesH,
                           double scalesW, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleTrilinear3d,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleTrilinear3dBackwardGetWorkspaceSize,
                           const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize, bool alignCorners,
                           double scalesD, double scalesH, double scalesW, aclTensor* gradInput, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnUpsampleTrilinear3dBackward,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnWeightQuantBatchMatmulGetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* diagonalMatrix, const aclTensor* deqOffset,
                           const aclTensor* deqScale, const aclTensor* addOffset, const aclTensor* mulScale, const aclTensor* bias,
                           bool transposeX1, bool transposeX2, float antiquantScale, float antiquantOffset, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnWeightQuantBatchMatmul,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnWeightQuantBatchMatmulNzGetWorkspaceSize,
                           const aclTensor* x, const aclTensor* weight, const aclTensor* antiquantScale,
                           const aclTensor* antiquantOffsetOptional, const aclTensor* quantScaleOptional, const aclTensor* quantOffsetOptional,
                           const aclTensor* biasOptional, int antiquantGroupSize, const aclTensor* y, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnWeightQuantBatchMatmulNz,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnWeightQuantBatchMatmulV2GetWorkspaceSize,
                           const aclTensor* x, const aclTensor* weight, const aclTensor* antiquantScale,
                           const aclTensor* antiquantOffsetOptional, const aclTensor* quantScaleOptional, const aclTensor* quantOffsetOptional,
                           const aclTensor* biasOptional, int antiquantGroupSize, const aclTensor* y, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnWeightQuantBatchMatmulV2,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnWeightQuantBatchMatmulV3GetWorkspaceSize,
                           const aclTensor* x, const aclTensor* weight, const aclTensor* antiquantScale,
                           const aclTensor* antiquantOffsetOptional, const aclTensor* quantScaleOptional, const aclTensor* quantOffsetOptional,
                           const aclTensor* biasOptional, int antiquantGroupSize, int innerPrecise, const aclTensor* y,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnWeightQuantBatchMatmulV3,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnWeightQuantMatmulAllReduceGetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* antiquantScale,
                           const aclTensor* antiquantOffset, const aclTensor* x3, const char* group, const char* reduceOp, int64_t commTurn,
                           int64_t streamMode, int64_t antiquantGroupSize, const aclTensor* output, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnWeightQuantMatmulAllReduce,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize,
                           const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* antiquantScale,
                           const aclTensor* antiquantOffset, const aclTensor* residual, const aclTensor* gamma, double epsilon,
                           const char* group, const char* reduceOp, int64_t commTurn, int64_t streamMode, int64_t antiquantGroupSize,
                           const aclTensor* y, const aclTensor* normOut, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnWeightQuantMatmulAllReduceAddRmsNorm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnXLogYScalarOtherGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnXLogYScalarOther,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnXLogYScalarSelfGetWorkspaceSize,
                           const aclScalar* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnXLogYScalarSelf,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnXLogYTensorGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* other, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnXLogYTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)


#ifdef __cplusplus
}
#endif
