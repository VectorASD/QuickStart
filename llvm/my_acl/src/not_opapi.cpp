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
    // На примере test_unary_pointwise_ops.py:
 
 /* auto gen = at::detail::getDefaultCPUGenerator();
    gen.set_current_seed(static_cast<uint64_t>(seed) + static_cast<uint64_t>(offset));
    selfRef.normal_(mean, std, gen);

    2465 passed, 6 skipped, 1 warning in 99.33s (0:01:39)
    aclnnInplaceNormal:           count=2591 | min=  5.143 us | avg=21251.568 us | max=159883.171 us | sum=55062813.281 us */

 /* selfRef.zero_();

    80 failed, 2385 passed, 6 skipped, 1 warning in 51.20s
    aclnnInplaceNormal:           count= 2591 | min=  3.401 us | avg= 1204.005 us | max= 17830.911 us | sum=3119576.408 us */
    
    // Вывод: 55062813.281 us - 3119576.408 us = 51.94s из 99.33s (52%) всего времени улетает
    // исключительно на работу этой функции, исключая работы malloc/free за пределами данной операции!

 /* std::ostringstream oss;
    oss << "RANDOM: mean: " << mean << ", std: " << stdd << ", offset: " << offset << ", seed: " << seed;
    log_output(oss, true);
        Хороший показательный пример того, что seed по умолчанию НИКОГДА не меняется! Только offset :)
        Т.е. это лишь подкрепляет мою личную идею с кешированием рандомных чисел */

    cached_normal_(selfRef, seed, offset, mean, std);
    // aclnnInplaceNormal:           count=2583 | min=  6.232 us | avg=1618.561 us | max=53768.063 us | sum=4832651.754 us
    // 2465 passed, 6 skipped, 1 warning in 45.12s (0:01:39)

    // После того, как test_accuracy_tile и test_accuracy_repeat были ограничены на 100 Mb (от этого же растёт ещё и кеш рандома)
    // Бонусом, кеш расширяется, не пересчитывая весь тензор, а только недостающую для расширения часть
    // aclnnInplaceNormal:           count=2583 | min=  7.450 us | avg= 1654.452 us | max=87228.201 us | sum=4273448.321 us
    // 2457 passed, 26 skipped, 1 warning in 38.73s
})
MAKE_OP(aclnnInplaceNormalTensor(out const aclTensor* selfRef, float mean, float std,
                                 const aclTensor* seedTensor, const aclTensor* offsetTensor, int64_t offset,
                                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    int64_t base_seed = seedTensor.item<int64_t>() + offsetTensor.item<int64_t>();
    cached_normal_(selfRef, base_seed, offset, mean, std);
})

MAKE_OP(aclnnInplaceRandom(out const aclTensor* selfRef, int64_t from, int64_t to, int64_t seed, int64_t offset,
                           uint64_t* workspaceSize, aclOpExecutor** executor) {
    cached_random_(selfRef, seed, offset, from, to);
})
MAKE_OP(aclnnInplaceRandomTensor(out const aclTensor* selfRef, int64_t from, int64_t to,
                                 const aclTensor* seedTensor, const aclTensor* offsetTensor, int64_t offset,
                                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    int64_t base_seed = seedTensor.item<int64_t>() + offsetTensor.item<int64_t>();
    cached_random_(selfRef, base_seed, offset, from, to);
})

MAKE_OP(aclnnInplaceUniform(out const aclTensor* selfRef, double from, double to, uint64_t seed, uint64_t offset,
                           uint64_t* workspaceSize, aclOpExecutor** executor) {
    cached_uniform_(selfRef, static_cast<int64_t>(seed), static_cast<int64_t>(offset), from, to);
})
MAKE_OP(aclnnInplaceUniformTensor(out const aclTensor* selfRef, double from, double to,
                                 const aclTensor* seedTensor, const aclTensor* offsetTensor,
                                 uint64_t offset, uint64_t* workspaceSize, aclOpExecutor** executor) {
    int64_t base_seed = static_cast<int64_t>(seedTensor.item<int64_t>()) + static_cast<int64_t>(offsetTensor.item<int64_t>());
    cached_uniform_(selfRef, base_seed, static_cast<int64_t>(offset), from, to);
})

MAKE_OP(aclnnRandperm(int64_t n, int64_t seed, int64_t offset, sync aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    // Невозможно создать такой кеш, чтобы не нарушить правило перестановки
    auto gen = at::detail::getDefaultCPUGenerator();
    gen.set_current_seed(static_cast<uint64_t>(seed) + static_cast<uint64_t>(offset));
    at::randperm_out(out, n, gen);
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
    selfRef.fill_diagonal_(fillValue, wrap);
})
MAKE_OP(aclnnInplaceFillScalar(out aclTensor* selfRef, const aclScalar* value,
                                uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.fill_(value);
})
MAKE_OP(aclnnInplaceFillTensor(out aclTensor* selfRef, const aclTensor* value,
                                uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.copy_(value);
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

MAKE_OP(aclnnRange(const aclScalar* start, const aclScalar* end, const aclScalar* step,
                   out aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    out.copy_(at::range(_start, _end, step, out.options()));
})
MAKE_OP(aclnnArange(const aclScalar* start, const aclScalar* end, const aclScalar* step,
                    out aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    out.copy_(at::arange(_start, _end, step, at::device(at::kCPU).dtype(out.scalar_type())));
})
MAKE_OP(aclnnLinspace(const aclScalar* start, const aclScalar* end, int64_t steps,
                      out aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    out.copy_(at::linspace(_start, _end, steps, out.options()));
})
MAKE_OP(aclnnLogSpace(const aclScalar* start, const aclScalar* end, int64_t steps, double base,
                      out aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::logspace_out(out, _start, _end, steps, base);
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
MAKE_OP(aclnnInplaceCeil(out aclTensor* selfRef,
                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.ceil_();
})

MAKE_OP(aclnnIsFinite(const aclTensor* self, out aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    out.copy_(at::isfinite(self));
})

MAKE_OP(aclnnAngleV2(const aclTensor* x, out const aclTensor* out,
                     uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::angle_out(out, x);
})

MAKE_OP(aclnnLeftShift(const aclTensor* self, const aclTensor* shiftBits, out aclTensor* out,
                       uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::bitwise_left_shift_out(out, self, shiftBits);
})
MAKE_OP(aclnnLeftShifts(const aclTensor* self, const aclScalar* shiftBits, out aclTensor* out,
                        uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::bitwise_left_shift_out(out, self, shiftBits);
})

MAKE_OP(aclnnRightShift(const aclTensor* input, const aclTensor* shiftBits,
                        out aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::bitwise_right_shift_out(out, input, shiftBits);
})


MAKE_OP(aclnnSin(const aclTensor* self, out aclTensor* out,
                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::sin_out(out, self);
})
MAKE_OP(aclnnInplaceSin(out aclTensor* selfRef,
                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.sin_();
})

MAKE_OP(aclnnSinc(const aclTensor* self, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::sinc_out(out, self);
})
MAKE_OP(aclnnInplaceSinc(out aclTensor* selfRef,
                          uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.sinc_();
})

MAKE_OP(aclnnSinh(const aclTensor* self, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::sinh_out(out, self);
})
MAKE_OP(aclnnInplaceSinh(out aclTensor* selfRef,
                          uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.sinh_();
})


MAKE_OP(aclnnAsin(const aclTensor* self, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::asin_out(out, self);
})
MAKE_OP(aclnnInplaceAsin(out aclTensor* selfRef,
                          uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.asin_();
})

MAKE_OP(aclnnAsinh(const aclTensor* input, out aclTensor* out,
                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::asinh_out(out, input);
})
MAKE_OP(aclnnInplaceAsinh(out aclTensor* inputRef,
                           uint64_t* workspaceSize, aclOpExecutor** executor) {
    inputRef.asinh_();
})


MAKE_OP(aclnnCos(const aclTensor* input, out aclTensor* out,
                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::cos_out(out, input);
})
MAKE_OP(aclnnInplaceCos(out aclTensor* inputRef, uint64_t* workspaceSize, aclOpExecutor** executor) {
    inputRef.cos_();
})

MAKE_OP(aclnnCosh(const aclTensor* self, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::cosh_out(out, self);
})
MAKE_OP(aclnnInplaceCosh(out aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.cosh_();
})


MAKE_OP(aclnnAcos(const aclTensor* input, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::acos_out(out, input);
})
MAKE_OP(aclnnInplaceAcos(out aclTensor* inputRef, uint64_t* workspaceSize, aclOpExecutor** executor) {
    inputRef.acos_();
})

MAKE_OP(aclnnAcosh(const aclTensor* self, out aclTensor* out,
                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::acosh_out(out, self);
})
MAKE_OP(aclnnInplaceAcosh(out aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.acosh_();
})


MAKE_OP(aclnnTan(const aclTensor* self, out aclTensor* out,
                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::tan_out(out, self);
})
MAKE_OP(aclnnInplaceTan(out const aclTensor* selfRef,
                        uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.tan_();
})

MAKE_OP(aclnnAtan(const aclTensor* input, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::atan_out(out, input);
})
MAKE_OP(aclnnInplaceAtan(out aclTensor* inputRef,
                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    inputRef.atan_();
})


MAKE_OP(aclnnExp(const aclTensor* self, out aclTensor* out,
                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::exp_out(out, self);
})
MAKE_OP(aclnnInplaceExp(out aclTensor* selfRef,
                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.exp_();
})

MAKE_OP(aclnnExpm1(const aclTensor* self, out aclTensor* out,
                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::expm1_out(out, self);
})
MAKE_OP(aclnnInplaceExpm1(out aclTensor* selfRef,
                           uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.expm1_();
})

MAKE_OP(aclnnExp2(const aclTensor* self, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::exp2_out(out, self);
})
MAKE_OP(aclnnInplaceExp2(out aclTensor* selfRef,
                          uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.copy_(at::exp2(selfRef));
})


MAKE_OP(aclnnNeg(const aclTensor* self, out aclTensor* out,
                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::neg_out(out, self);
})
MAKE_OP(aclnnInplaceNeg(out aclTensor* selfRef,
                        uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.neg_();
})


MAKE_OP(aclnnReciprocal(const aclTensor* self, out aclTensor* out,
                        uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::reciprocal_out(out, self);
})
MAKE_OP(aclnnInplaceReciprocal(out const aclTensor* selfRef,
                               uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.reciprocal_();
})


MAKE_OP(aclnnRsqrt(const aclTensor* self, out aclTensor* out,
                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::rsqrt_out(out, self);
})
MAKE_OP(aclnnInplaceRsqrt(out aclTensor* selfRef,
                          uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.rsqrt_();
})


MAKE_OP(aclnnFlip(const aclTensor* self, const aclIntArray* dims, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::flip_out(out, self, dims);
})


MAKE_OP(aclnnRepeat(const aclTensor* self, const aclIntArray* repeats, sync aclTensor* out,
                    uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::repeat_out(out, self, repeats);
})
MAKE_OP(aclnnRepeatInterleave(const aclTensor* self, const aclTensor* repeats, int64_t outputSize,
                              sync aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto output_size = outputSize > 0 ? std::optional<int64_t>(outputSize) : std::nullopt;
    out.copy_(at::repeat_interleave(self, repeats, /*dim=*/std::nullopt, output_size));
})
MAKE_OP(aclnnRepeatInterleaveInt(const aclTensor* self, int64_t repeats, int64_t outputSize,
                                 sync aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto output_size = outputSize > 0 ? std::optional<int64_t>(outputSize) : std::nullopt;
    out.copy_(at::repeat_interleave(self, repeats, /*dim=*/std::nullopt, output_size));
})
MAKE_OP(aclnnRepeatInterleaveIntWithDim(const aclTensor* self, int64_t repeats, int64_t dim,
                                        int64_t outputSize, sync aclTensor* out, uint64_t* workspaceSize,
                                        aclOpExecutor** executor) {
    auto output_size = outputSize > 0 ? std::optional<int64_t>(outputSize) : std::nullopt;
    out.copy_(at::repeat_interleave(self, repeats, dim, output_size));
})
MAKE_OP(aclnnRepeatInterleaveWithDim(const aclTensor* self, const aclTensor* repeats, int64_t dim,
                                     int64_t outputSize, sync aclTensor* out, uint64_t* workspaceSize,
                                     aclOpExecutor** executor) {
    auto output_size = outputSize > 0 ? std::optional<int64_t>(outputSize) : std::nullopt;
    out.copy_(at::repeat_interleave(self, repeats, dim, output_size));
})
MAKE_OP(aclnnRepeatInterleaveTensor(const aclTensor* repeats, int64_t outputSize,
                                    sync aclTensor* out, uint64_t* workspaceSize,
                                    aclOpExecutor** executor) {
    // repeats – 1D тензор с числом повторений для каждого индекса
    auto r = repeats;
    int64_t num_elements = r.numel();
    // Генерируем последовательность [0, 1, 2, ..., num_elements-1]
    auto indices = at::arange(num_elements, at::device(at::kCPU).dtype(at::kLong));
    // Повторяем индексы согласно repeats, общая длина = outputSize
    auto output_size = outputSize > 0 ? std::optional<int64_t>(outputSize) : std::nullopt;
    auto result = at::repeat_interleave(indices, r, output_size);
    out.copy_(result);
})


MAKE_OP(aclnnLog(const aclTensor* self, out aclTensor* out,
                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::log_out(out, self);
})
MAKE_OP(aclnnInplaceLog(out aclTensor* selfRef,
                        uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.log_();
})

MAKE_OP(aclnnLog10(const aclTensor* self, out aclTensor* out,
                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::log10_out(out, self);
})
MAKE_OP(aclnnInplaceLog10(out aclTensor* selfRef,
                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.log10_();
})

MAKE_OP(aclnnLog1p(const aclTensor* self, out aclTensor* out,
                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::log1p_out(out, self);
})
MAKE_OP(aclnnInplaceLog1p(out aclTensor* selfRef,
                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.log1p_();
})

MAKE_OP(aclnnLog2(const aclTensor* self, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::log2_out(out, self);
})
MAKE_OP(aclnnInplaceLog2(out aclTensor* selfRef,
                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.log2_();
})

MAKE_OP(aclnnLogdet(const aclTensor* self, out aclTensor* out,
                    uint64_t* workspaceSize, aclOpExecutor** executor) {
    out.copy_(at::logdet(self));
})


MAKE_OP(aclnnSqrt(const aclTensor* self, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::sqrt_out(out, self);
})
MAKE_OP(aclnnInplaceSqrt(out aclTensor* self,
                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    self.sqrt_();
})


MAKE_OP(aclnnNonzero(const aclTensor* self, sync aclTensor* out,
                    uint64_t* workspaceSize, aclOpExecutor** executor) {
    exec->out->store(at::nonzero(self));
})
MAKE_OP(aclnnNonzeroV2(const aclTensor* self, sync aclTensor* out,
                       uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto idx = at::nonzero(self);           // [N, ndim]
    exec->out->store(idx.transpose(0, 1));  // [ndim, N]
})


// Функции активаций и потерь

MAKE_OP(aclnnFastGelu(const aclTensor* self, out aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::gelu_out(out, self, "tanh");
})
MAKE_OP(aclnnFastGeluBackward(const aclTensor* gradOutput, const aclTensor* self, out aclTensor* gradInput,
                              uint64_t* workspaceSize, aclOpExecutor** executor) {
    gradInput.copy_(at::gelu_backward(gradOutput, self, "tanh"));
})

MAKE_OP(aclnnGelu(const aclTensor* self, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::gelu_out(out, self);
})
MAKE_OP(aclnnGeluV2(const aclTensor* x, int64_t approximate, out aclTensor* y,
                    uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::gelu_out(y, x, approximate ? "tanh" : "none");
})

MAKE_OP(aclnnGeluBackward(const aclTensor* gradOutput, const aclTensor* self, out aclTensor* gradInput,
                          uint64_t* workspaceSize, aclOpExecutor** executor) {
    gradInput.copy_(at::gelu_backward(gradOutput, self));
})
MAKE_OP(aclnnGeluBackwardV2(const aclTensor* gradOutput, const aclTensor* self, char* approximate, out aclTensor* gradInput,
                            uint64_t* workspaceSize, aclOpExecutor** executor) {
    bool tanh = (approximate == "tanh");
    gradInput.copy_(at::gelu_backward(gradOutput, self, tanh ? "tanh" : "none"));
    // pytorch/third_party/op-plugin/op_plugin/utils/KernelNpuNewParams.cpp:26
    // уже есть проверка на то, что это "tanh" или "none"
})


MAKE_OP(aclnnGlu(const aclTensor* self, int64_t dim, sync aclTensor* out,
                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    // auto chunks = self.chunk(2, dim);
    // out.copy_(chunks[0] * at::sigmoid(chunks[1]));
    out.copy_(at::glu(self, dim));
})
MAKE_OP(aclnnGluBackward(const aclTensor* gradOut, const aclTensor* self, int64_t dim,
                         out aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    out.copy_(at::glu_backward(gradOut, self, dim));
})


MAKE_OP(aclnnSwiGlu(const aclTensor* self, int64_t dim, sync aclTensor* out,
                    uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto chunks = self.chunk(2, dim);
    auto a = chunks[0], b = chunks[1];
    out.copy_(at::silu(a) * b);
})
MAKE_OP(aclnnSwiGluGrad(const aclTensor* gradOut, const aclTensor* self, int64_t dim,
                        out aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto orig_dtype = gradOut.scalar_type();
    bool need_cast = (orig_dtype == at::kHalf || orig_dtype == at::kBFloat16);
    if (need_cast)
        self = self.to(at::kFloat);

    auto chunks = self.chunk(2, dim);
    auto a = chunks[0], b = chunks[1];
    auto sig_a = at::sigmoid(a);
    auto silu_a = a * sig_a;  // silu(a)
    auto d_silu = sig_a * (1.0 + a * (1.0 - sig_a));  // производная silu(a) по a

    auto grad_a = gradOut * b * d_silu;
    auto grad_b = gradOut * silu_a;
    auto result = at::cat({grad_a, grad_b}, dim);
    out.copy_(result);  // автоматические приводит тип result (Float) к типу out (Half, BFloat16)
})


MAKE_OP(aclnnGeGlu(const aclTensor* self, int64_t dim, int64_t approximate,
                   sync aclTensor* out, out aclTensor* outGelu,
                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto chunks = self.chunk(2, dim);
    auto a = chunks[0], b = chunks[1];
    at::Tensor gelu_a = at::gelu(a, approximate ? "tanh" : "none");
    out.copy_(gelu_a * b);
    outGelu.copy_(gelu_a);
})
MAKE_OP(aclnnGeGluBackward(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gelu,
                           int64_t dim, int64_t approximate, out aclTensor* gradInput,
                           uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto chunks = self.chunk(2, dim);
    auto a = chunks[0], b = chunks[1];
    at::Tensor grad_a = at::gelu_backward(gradOutput * b, a, approximate ? "tanh" : "none");
    at::Tensor grad_b = gradOutput * gelu;
    gradInput.copy_(at::cat({grad_a, grad_b}, dim));
})
MAKE_OP(aclnnGeGluV3(const aclTensor* self, int64_t dim, int64_t approximate, bool activateLeft,
                     sync aclTensor* out, out aclTensor* outGelu,
                     uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto chunks = self.chunk(2, dim);
    auto a = activateLeft ? chunks[0] : chunks[1];
    auto b = activateLeft ? chunks[1] : chunks[0];
    at::Tensor gelu_a = at::gelu(a, approximate ? "tanh" : "none");
    out.copy_(gelu_a * b);
    outGelu.copy_(gelu_a);
})
MAKE_OP(aclnnGeGluV3Backward(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gelu,
                             int64_t dim, int64_t approximate, bool activateLeft, out aclTensor* gradInput,
                             uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto chunks = self.chunk(2, dim);
    auto a = activateLeft ? chunks[0] : chunks[1];
    auto b = activateLeft ? chunks[1] : chunks[0];
    at::Tensor grad_a = at::gelu_backward(gradOutput * b, a, (approximate == 1) ? "tanh" : "none");
    at::Tensor grad_b = gradOutput * gelu;
    gradInput.copy_(at::cat(activateLeft ? std::vector<at::Tensor>{grad_a, grad_b}
                                         : std::vector<at::Tensor>{grad_b, grad_a}, dim));
})


MAKE_OP(aclnnElu(const aclTensor* self, const aclScalar* alpha, const aclScalar* scale,
                 const aclScalar* inputScale, sync aclTensor* out,
                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::elu_out(out, self, alpha, scale, inputScale);
})
MAKE_OP(aclnnEluBackward(const aclTensor* gradOutput, const aclScalar* alpha,
                         const aclScalar* scale, const aclScalar* inputScale,
                         bool isResult, const aclTensor* selfOrResult,
                         out aclTensor* gradInput, uint64_t* workspaceSize,
                         aclOpExecutor** executor) {
    gradInput.copy_(at::elu_backward(gradOutput, alpha, scale, inputScale, isResult, selfOrResult));
})
MAKE_OP(aclnnInplaceElu(out aclTensor* selfRef, const aclScalar* alpha,
                        const aclScalar* scale, const aclScalar* inputScale,
                        uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::elu_(selfRef, alpha, scale, inputScale);
})


MAKE_OP(aclnnCelu(const aclTensor* self, const aclScalar* alpha, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::celu_out(out, self, alpha);
})
MAKE_OP(aclnnInplaceCelu(out aclTensor* selfRef, const aclScalar* alpha,
                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::celu_(selfRef, alpha);
})


MAKE_OP(aclnnRelu(const aclTensor* self, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::relu_out(out, self);
})
MAKE_OP(aclnnInplaceRelu(out aclTensor* selfRef,
                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::relu_(selfRef);
})


MAKE_OP(aclnnLeakyRelu(const aclTensor* self, const aclScalar* negativeSlope,
                       out aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::leaky_relu_out(out, self, negativeSlope);
})
MAKE_OP(aclnnInplaceLeakyRelu(out aclTensor* selfRef, const aclScalar* negativeSlope,
                              uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::leaky_relu_(selfRef, negativeSlope);
})
MAKE_OP(aclnnLeakyReluBackward(const aclTensor* gradOutput, const aclTensor* self,
                               const aclScalar* negativeSlope, bool selfIsResult,
                               out aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    out.copy_(at::leaky_relu_backward(gradOutput, self, negativeSlope, selfIsResult));
})


MAKE_OP(aclnnRReluWithNoise(const aclTensor* self, const aclTensor* noise,
                            const aclScalar* lower, const aclScalar* upper,
                            bool training, int64_t seed, int64_t offset,
                            out aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto gen = at::detail::getDefaultCPUGenerator();
    gen.set_current_seed(static_cast<uint64_t>(seed + offset));
    at::rrelu_with_noise_out(out, self, noise, lower, upper, training, gen);
})
MAKE_OP(aclnnInplaceRReluWithNoise(out aclTensor* selfRef, const aclTensor* noise,
                                   const aclScalar* lower, const aclScalar* upper,
                                   bool training, int64_t seed, int64_t offset,
                                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto gen = at::detail::getDefaultCPUGenerator();
    gen.set_current_seed(static_cast<uint64_t>(seed + offset));
    at::rrelu_with_noise_(selfRef, noise, lower, upper, training, gen);
})


MAKE_OP(aclnnSoftMarginLoss(const aclTensor* self, const aclTensor* target,
                            int64_t reduction, sync aclTensor* out,
                            uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::soft_margin_loss_out(out, self, target, reduction);
})
MAKE_OP(aclnnSoftMarginLossBackward(const aclTensor* gradOutput, const aclTensor* self,
                                    const aclTensor* target, int64_t reduction,
                                    out aclTensor* out, uint64_t* workspaceSize,
                                    aclOpExecutor** executor) {
    out.copy_(at::soft_margin_loss_backward(gradOutput, self, target, reduction));
})


MAKE_OP(aclnnSoftmax(const aclTensor* self, int64_t dim, out aclTensor* out,
                     uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::softmax_out(out, self, dim);
})
MAKE_OP(aclnnSoftmaxBackward(const aclTensor* gradOutput, const aclTensor* output,
                             int64_t dim, out aclTensor* out, uint64_t* workspaceSize,
                             aclOpExecutor** executor) {
    at::_softmax_backward_data_out(out, gradOutput, output, dim, output.scalar_type());
})


MAKE_OP(aclnnSoftplus(const aclTensor* self, const aclScalar* beta,
                      const aclScalar* threshold, out aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::softplus_out(out, self, beta, threshold);
})
MAKE_OP(aclnnSoftplusBackward(const aclTensor* gradOutput, const aclTensor* self,
                              const aclScalar* beta, const aclScalar* threshold,
                              out aclTensor* gradInput, uint64_t* workspaceSize,
                              aclOpExecutor** executor) {
    at::softplus_backward_out(gradInput, gradOutput, self, beta, threshold);
})


MAKE_OP(aclnnSoftshrink(const aclTensor* self, const aclScalar* lambd, out aclTensor* out,
                        uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::softshrink_out(out, self, lambd);
})
MAKE_OP(aclnnSoftshrinkBackward(const aclTensor* gradOutput, const aclTensor* self,
                                const aclScalar* lambda, out aclTensor* gradInput,
                                uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::softshrink_backward_out(gradInput, gradOutput, self, lambda);
})


MAKE_OP(aclnnBinaryCrossEntropy(const aclTensor* self, const aclTensor* target,
                                const aclTensor* weight, int64_t reduction,
                                sync aclTensor* out, uint64_t* workspaceSize,
                                aclOpExecutor** executor) {
    at::binary_cross_entropy_out(out, self, target, weight, reduction);
})
MAKE_OP(aclnnBinaryCrossEntropyBackward(const aclTensor* gradOutput, const aclTensor* self,
                                        const aclTensor* target, const aclTensor* weightOptional,
                                        int64_t reduction, out aclTensor* out,
                                        uint64_t* workspaceSize, aclOpExecutor** executor) {
    out.copy_(at::binary_cross_entropy_backward(gradOutput, self, target, weightOptional, reduction));
})


MAKE_OP(aclnnBinaryCrossEntropyWithLogits(const aclTensor* self, const aclTensor* target,
                                          const aclTensor* weightOptional,
                                          const aclTensor* posWeightOptional,
                                          int64_t reduction, sync aclTensor* out,
                                          uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::binary_cross_entropy_with_logits_out(out, self, target, weightOptional, posWeightOptional, reduction);
})
MAKE_OP(aclnnBinaryCrossEntropyWithLogitsBackward(const aclTensor* gradOutput, const aclTensor* self,
                                                  const aclTensor* target, const aclTensor* weightOptional,
                                                  const aclTensor* posWeightOptional, int64_t reduction,
                                                  out aclTensor* out, uint64_t* workspaceSize,
                                                  aclOpExecutor** executor) {
    auto sig = at::sigmoid(self);
    // Если weight не задан, используем 1; если pos_weight не задан, используем 1.
    auto weight = weightOptional.defined() ? weightOptional : at::ones_like(self);
    auto pos_weight = posWeightOptional.defined() ? posWeightOptional : at::ones_like(self);
    // Формула градиента BCEWithLogitsLoss:
    // dl/dx = (sigmoid(x) * (1 + target * (pos_weight - 1)) - pos_weight * target) * weight
    auto grad = (sig * (1.0 + target * (pos_weight - 1.0)) - pos_weight * target) * weight;
    if (reduction == at::Reduction::Mean) grad = grad / self.numel();
    // Для Reduction::Sum оставляем grad как есть
    out.copy_(gradOutput * grad);
})


// ---------- BinaryCrossEntropyWithLogitsTargetBackward ----------
// Градиент BCEWithLogitsLoss по целевому тензору (target).
// Формула: dl/dp = -weight * (pos_weight * log(sigmoid(x)) - log(1 - sigmoid(x)))
// Если pos_weight == nullptr, то dl/dp = -weight * x
MAKE_OP(aclnnBinaryCrossEntropyWithLogitsTargetBackward(const aclTensor* gradOutput, const aclTensor* self,
                                                        const aclTensor* target,
                                                        const aclTensor* weightOptional,
                                                        const aclTensor* posWeightOptional,
                                                        int64_t reduction, out aclTensor* gradTarget,
                                                        uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto x = self;

    at::Tensor grad_p;
    if (posWeightOptional.defined()) {
        auto sig = at::sigmoid(x);
        auto log_sig = at::log(sig);
        auto log_1m_sig = at::log(1.0 - sig);
        grad_p = -(weightOptional * (posWeightOptional * log_sig - log_1m_sig));
    } else
        grad_p = -weightOptional * x;

    if (reduction == at::Reduction::Mean) {
        grad_p = grad_p / x.numel();
    } // для Sum ничего не делаем
    gradTarget.copy_(gradOutput * grad_p);
})


MAKE_OP(aclnnSoftmaxCrossEntropyWithLogits(const aclTensor* features, const aclTensor* labels,
                                           sync aclTensor* loss, sync aclTensor* backprop,
                                           uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto log_softmax = at::log_softmax(features, -1);
    // weight = пустой тензор
    auto loss_t = at::nll_loss(log_softmax, labels.to(at::kLong), at::Tensor(), at::Reduction::None);
    loss.copy_(loss_t);

    auto softmax = at::softmax(features, -1);
    int64_t num_classes = features.size(-1);
    auto one_hot = at::one_hot(labels.to(at::kLong), num_classes).to(features.scalar_type());
    backprop.copy_(softmax - one_hot);
})


MAKE_OP(aclnnSigmoid(const aclTensor* self, out aclTensor* out,
                     uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::sigmoid_out(out, self);
})
MAKE_OP(aclnnInplaceSigmoid(out aclTensor* selfRef,
                            uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::sigmoid_(selfRef);
})
MAKE_OP(aclnnSigmoidBackward(const aclTensor* gradOutput, const aclTensor* output,
                             out aclTensor* gradInput, uint64_t* workspaceSize,
                             aclOpExecutor** executor) {
    at::sigmoid_backward_out(gradInput, gradOutput, output);
})

MAKE_OP(aclnnLogSigmoid(const aclTensor* self, out aclTensor* out,
                        uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::log_sigmoid_out(out, self);
})
MAKE_OP(aclnnLogSigmoidForward(const aclTensor* self, out aclTensor* out,
                               sync aclTensor* buffer, uint64_t* workspaceSize,
                               aclOpExecutor** executor) {
    auto result_buffer = at::empty_like(self);
    at::log_sigmoid_forward_out(out, result_buffer, self);
    exec->buffer->store(result_buffer);

    // store используется ТОЛЬКО в тех случаях,
    // когда может потребоваться БОЛЬШЕ места,
    // чем это УЖЕ выделено благодаря torch_npu

    // в остальных же случаях, если ДОСТАТОЧНО синхронизировать
    // shape, stride и offset выхода (а его тип - aclTensor)
    // с результатом at::Tensor, просто используем sync
})
MAKE_OP(aclnnLogSigmoidBackward(const aclTensor* gradOutput, const aclTensor* self,
                                const aclTensor* buffer, out aclTensor* gradInput,
                                uint64_t* workspaceSize, aclOpExecutor** executor) {
    gradInput.copy_(at::log_sigmoid_backward(gradOutput, self, buffer));
})


MAKE_OP(aclnnSelu(const aclTensor* self, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    out.copy_(at::selu(self));
})
MAKE_OP(aclnnInplaceSelu(out aclTensor* selfRef,
                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::selu_(selfRef);
})
MAKE_OP(aclnnSeluBackward(const aclTensor* gradOutput, const aclTensor* result,
                          out aclTensor* gradInput, uint64_t* workspaceSize,
                          aclOpExecutor** executor) {
    // at::selu_backward_out(gradInput, gradOutput, result);  OOPS, is not defined...
    const double alpha = 1.6732632423543772848170429916717;
    const double scale = 1.0507009873554804934193349852946;
    // SELU: y = scale * (max(0,x) + min(0, alpha * (exp(x) - 1)))
    // result — это выход прямого прохода (y)
    // Для x >= 0: y = scale * x   => dy/dx = scale
    // Для x < 0:  y = scale * alpha * (exp(x) - 1)
    // => x = ln(y / (scale * alpha) + 1)
    // => dy/dx = scale * alpha * exp(x) = y + scale * alpha
    auto grad = at::where(result >= 0.0, scale, result + scale * alpha);
    gradInput.copy_(gradOutput * grad);
})


MAKE_OP(aclnnSilu(const aclTensor* self, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::silu_out(out, self);
})
MAKE_OP(aclnnInplaceSilu(out aclTensor* selfRef,
                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::silu_out(selfRef, selfRef);
})
MAKE_OP(aclnnSiluBackward(const aclTensor* gradOutput, const aclTensor* self,
                          out aclTensor* gradInput, uint64_t* workspaceSize,
                          aclOpExecutor** executor) {
    at::silu_backward_out(gradInput, gradOutput, self);
})


MAKE_OP(aclnnTanh(const aclTensor* self, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::tanh_out(out, self);
})
MAKE_OP(aclnnTanhBackward(const aclTensor* gradOutput, const aclTensor* output, out aclTensor* gradInput,
                          uint64_t* workspaceSize, aclOpExecutor** executor) {
    gradInput.copy_(at::tanh_backward(gradOutput, output));
})
MAKE_OP(aclnnInplaceTanh(out aclTensor* selfRef,
                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.tanh_();
})

MAKE_OP(aclnnAtanh(const aclTensor* input, out aclTensor* out,
                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::atanh_out(out, input);
})
MAKE_OP(aclnnInplaceAtanh(out aclTensor* inputRef,
                          uint64_t* workspaceSize, aclOpExecutor** executor) {
    inputRef.atanh_();
})


MAKE_OP(aclnnErf(const aclTensor* self, out aclTensor* out,
                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::erf_out(out, self);
})
MAKE_OP(aclnnInplaceErf(out aclTensor* selfRef,
                        uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.erf_();
})

MAKE_OP(aclnnErfc(const aclTensor* self, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::erfc_out(out, self);
})
MAKE_OP(aclnnInplaceErfc(out aclTensor* selfRef,
                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.erfc_();
})

MAKE_OP(aclnnErfinv(const aclTensor* self, out aclTensor* out,
                    uint64_t* workspaceSize, aclOpExecutor** executor) {
    out.copy_(at::erfinv(self));
})
MAKE_OP(aclnnInplaceErfinv(out aclTensor* selfRef,
                           uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.copy_(at::erfinv(selfRef));
})


MAKE_OP(aclnnTriu(const aclTensor* self, int64_t diagonal, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::triu_out(out, self, diagonal);
})
MAKE_OP(aclnnInplaceTriu(out aclTensor* selfRef, int64_t diagonal,
                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.triu_(diagonal);
})


MAKE_OP(aclnnTril(const aclTensor* self, int64_t diagonal, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::tril_out(out, self, diagonal);
})
MAKE_OP(aclnnInplaceTril(out aclTensor* selfRef, int64_t diagonal,
                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.tril_(diagonal);
})


MAKE_OP(aclnnLogSoftmax(const aclTensor* self, int64_t dim, out aclTensor* out,
                        uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::log_softmax_out(out, self, dim);
})
MAKE_OP(aclnnLogSoftmaxBackward(const aclTensor* gradOutput, const aclTensor* output,
                                int64_t dim, out aclTensor* out, uint64_t* workspaceSize,
                                aclOpExecutor** executor) {
    at::_log_softmax_backward_data_out(out, gradOutput, output, dim, output.scalar_type());
})


MAKE_OP(aclnnThreshold(const aclTensor* self, const aclScalar* threshold,
                       const aclScalar* value, out aclTensor* out,
                       uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::threshold_out(out, self, threshold, value);
})
MAKE_OP(aclnnInplaceThreshold(out aclTensor* selfRef, const aclScalar* threshold,
                              const aclScalar* value,
                              uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::threshold_(selfRef, threshold, value);
})
MAKE_OP(aclnnThresholdBackward(const aclTensor* gradOutput, const aclTensor* self,
                               const aclScalar* threshold, out aclTensor* out,
                               uint64_t* workspaceSize, aclOpExecutor** executor) {
    out.copy_(at::threshold_backward(gradOutput, self, threshold));
})


MAKE_OP(aclnnClampMax(const aclTensor* self,
                      const aclScalar* clipValueMax, out aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::clamp_max_out(out, self, clipValueMax);
})
MAKE_OP(aclnnInplaceClampMax(out aclTensor* selfRef,
                             const aclScalar* clipValueMax,
                             uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.clamp_max_(clipValueMax);
})

MAKE_OP(aclnnClampMaxTensor(const aclTensor* self, const aclTensor* max, out aclTensor* out,
                            uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::clamp_max_out(out, self, max);
})
MAKE_OP(aclnnInplaceClampMaxTensor(out aclTensor* selfRef, const aclTensor* max,
                                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.clamp_max_(max);
})

MAKE_OP(aclnnClampMin(const aclTensor* self,
                      const aclScalar* clipValueMin, out aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::clamp_min_out(out, self, clipValueMin);
})
MAKE_OP(aclnnClampMinTensor(const aclTensor* self, const aclTensor* clipValueMin, out aclTensor* out,
                            uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::clamp_min_out(out, self, clipValueMin);
})
MAKE_OP(aclnnInplaceClampMinTensor(out aclTensor* selfRef, const aclTensor* clipValueMin,
                                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.clamp_min_(clipValueMin);
})

MAKE_OP(aclnnClamp(const aclTensor* self,
                   optional const aclScalar* clipValueMin,
                   optional const aclScalar* clipValueMax,
                   out aclTensor* out,
                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::clamp_out(out, self, clipValueMin, clipValueMax);
})

MAKE_OP(aclnnClampTensor(const aclTensor* self,
                         optional const aclTensor* clipValueMin,
                         optional const aclTensor* clipValueMax,
                         out aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::clamp_out(out, self,
        clipValueMin.defined() ? std::optional<at::Tensor>(clipValueMin) : c10::nullopt,
        clipValueMax.defined() ? std::optional<at::Tensor>(clipValueMax) : c10::nullopt);
})


MAKE_OP(aclnnFloor(const aclTensor* self, out aclTensor* out,
                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::floor_out(out, self);
})
MAKE_OP(aclnnInplaceFloor(out aclTensor* selfRef,
                          uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.floor_();
})


MAKE_OP(aclnnNanToNum(const aclTensor* self, float nan, float posinf, float neginf, out aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::nan_to_num_out(out, self, nan, posinf, neginf);
})
MAKE_OP(aclnnInplaceNanToNum(out aclTensor* selfRef, float nan, float posinf, float neginf,
                             uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::nan_to_num_(selfRef, nan, posinf, neginf);
})

MAKE_OP(aclnnNanMedian(const aclTensor* self, out aclTensor* out,
                       uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::nanmedian_out(out, self);
})

MAKE_OP(aclnnNanMedianDim(const aclTensor* self, int64_t dim, bool keepDim,
                          out aclTensor* valuesOut, out aclTensor* indicesOut,
                          uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::nanmedian_out(valuesOut, indicesOut, self, dim, keepDim);
})


MAKE_OP(aclnnPolar(const aclTensor* input, const aclTensor* angle, out aclTensor* out,
                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::polar_out(out, input, angle);
})


MAKE_OP(aclnnLerp(const aclTensor* self, const aclTensor* end, const aclTensor* weight, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::lerp_out(out, self, _end, weight);
})
MAKE_OP(aclnnInplaceLerp(out aclTensor* selfRef, const aclTensor* end, const aclTensor* weight,
                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.lerp_(_end, weight);
})

MAKE_OP(aclnnLerps(const aclTensor* self, const aclTensor* end, const aclScalar* weight, out aclTensor* out,
                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::lerp_out(out, self, _end, weight);
})
MAKE_OP(aclnnInplaceLerps(out aclTensor* selfRef, const aclTensor* end, const aclScalar* weight,
                          uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.lerp_(_end, weight);
})


MAKE_OP(aclnnCumsum(const aclTensor* self, int64_t dim, aclDataType dtype, out aclTensor* out,
                    uint64_t* workspaceSize, aclOpExecutor** executor) {
    if (dtype && dtype != self.scalar_type()) {
        at::cumsum_out(out, self.to(dtype), dim);
    } else {
        at::cumsum_out(out, self, dim);
    }
})

MAKE_OP(aclnnCumsumV2(const aclTensor* self, int64_t dim, bool exclusive, bool reverse,
                      out aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::cumsum_out(out, self, dim);
    if (exclusive) {
        auto shifted = at::roll(out, 1, dim);
        shifted.select(dim, 0).fill_(0);
        out.copy_(shifted);
    }
    if (reverse) {
        out.copy_(at::flip(out, {dim}));
    }
})


MAKE_OP(aclnnProd(const aclTensor* self, aclDataType dtype, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    c10::optional<at::ScalarType> dt(dtype);
    out.copy_(at::prod(self, dt));
})

MAKE_OP(aclnnProdDim(const aclTensor* self, int64_t dim, bool keepDim, aclDataType dtype, out aclTensor* out,
                     uint64_t* workspaceSize, aclOpExecutor** executor) {
    c10::optional<at::ScalarType> dt(dtype);
    out.copy_(at::prod(self, dim, keepDim, dt));
})


MAKE_OP(aclnnGather(const aclTensor* self, int64_t dim, const aclTensor* index, out aclTensor* out,
                    uint64_t* workspaceSize, aclOpExecutor** executor) {
    out.copy_(at::gather(self, dim, index));
})

MAKE_OP(aclnnGatherV2(const aclTensor* self, int64_t dim, const aclTensor* index, out aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    out.copy_(at::gather(self, dim, index));
})


MAKE_OP(aclnnRound(const aclTensor* self, out aclTensor* out,
                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    out.copy_(at::round(self));
})

MAKE_OP(aclnnRoundDecimals(const aclTensor* self, int64_t decimals, out aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor) {
    out.copy_(at::round(self, decimals));
})
MAKE_OP(aclnnInplaceRound(aclTensor* selfRef,
                          uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.copy_(at::round(selfRef));
})

MAKE_OP(aclnnInplaceRoundDecimals(aclTensor* selfRef, int64_t decimals,
                                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.copy_(at::round(selfRef, decimals));
})


// ~~~ бинарные операции ~~~

MAKE_OP(aclnnAdd(const aclTensor* self, const aclTensor* other, const aclScalar* alpha, out aclTensor* out,
                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::add_out(out, self, other, alpha);
})
MAKE_OP(aclnnInplaceAdd(out aclTensor* selfRef, const aclTensor* other, const aclScalar* alpha,
                        uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.add_(other, alpha);
})

MAKE_OP(aclnnAddV3(const aclScalar* self, const aclTensor* other, const aclScalar* alpha, out aclTensor* out,
                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto self_t = at::scalar_tensor(self, other.options());
    at::add_out(out, self_t, other, alpha);
})
MAKE_OP(aclnnInplaceAddV3(out const aclScalar* selfRef, const aclTensor* other, const aclScalar* alpha,
                          uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto result = at::add(selfRef, other, alpha);
    selfRef.copy_(result);
})

MAKE_OP(aclnnAdds(const aclTensor* self, const aclScalar* other, const aclScalar* alpha, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::add_out(out, self, other, alpha);
})
MAKE_OP(aclnnInplaceAdds(out const aclTensor* selfRef, const aclScalar* other, const aclScalar* alpha,
                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.add_(other, alpha);
})


MAKE_OP(aclnnSub(const aclTensor* self, const aclTensor* other, const aclScalar* alpha, out aclTensor* out,
                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::sub_out(out, self, other, alpha);
})
MAKE_OP(aclnnInplaceSub(out aclTensor* selfRef, const aclTensor* other, const aclScalar* alpha,
                        uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.sub_(other, alpha);
})

MAKE_OP(aclnnSubs(const aclTensor* self, const aclScalar* other, const aclScalar* alpha, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::sub_out(out, self, other, alpha);
})
MAKE_OP(aclnnInplaceSubs(out aclTensor* selfRef, const aclScalar* other, const aclScalar* alpha,
                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.sub_(other, alpha);
})


MAKE_OP(aclnnRsub(const aclTensor* self, const aclTensor* other, const aclScalar* alpha, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    // out = other - alpha * self
    at::sub_out(out, other, self, alpha);
})
MAKE_OP(aclnnRsubs(const aclTensor* self, const aclScalar* other, const aclScalar* alpha, out aclTensor* out,
                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    // out = other - alpha * self
    auto other_t = at::scalar_tensor(other, self.options());
    at::sub_out(out, other_t, self, alpha);
})


MAKE_OP(aclnnMul(const aclTensor* self, const aclTensor* other, out aclTensor* out,
                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::mul_out(out, self, other);
})
MAKE_OP(aclnnInplaceMul(out aclTensor* selfRef, const aclTensor* other,
                        uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.mul_(other);
})

MAKE_OP(aclnnMuls(const aclTensor* self, const aclScalar* other, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    // other автоматически раскрывается в const at::Tensor&
    at::mul_out(out, self, other);
})
MAKE_OP(aclnnInplaceMuls(out aclTensor* selfRef, const aclScalar* other,
                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    // other автоматически раскрывается в const at::Tensor&
    selfRef.mul_(other);
})


MAKE_OP(aclnnDiv(const aclTensor* self, const aclTensor* other, out aclTensor* out,
                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::div_out(out, self, other);
})
MAKE_OP(aclnnInplaceDiv(out aclTensor* selfRef, const aclTensor* other,
                        uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.div_(other);
})

MAKE_OP(aclnnDivs(const aclTensor* self, const aclScalar* other, out aclTensor* out,
                  uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::div_out(out, self, other);
})
MAKE_OP(aclnnInplaceDivs(out aclTensor* selfRef, const aclScalar* other,
                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.div_(other);
})


MAKE_OP(aclnnDivMod(const aclTensor* self, const aclTensor* other, int mode, out aclTensor* out,
                    uint64_t* workspaceSize, aclOpExecutor** executor) {
    c10::optional<c10::string_view> rounding;
    switch (mode) {
        case 1:  rounding = "trunc"; break;
        case 2:  rounding = "floor"; break;
        default: rounding = c10::nullopt;
    }
    auto safe_other = at::where(other == 0, at::ones_like(other), other);
    at::div_out(out, self, safe_other, rounding);
    out.masked_fill_(other == 0, 0);
})
MAKE_OP(aclnnInplaceDivMod(out aclTensor* selfRef, const aclTensor* other, int mode,
                           uint64_t* workspaceSize, aclOpExecutor** executor) {
    c10::optional<c10::string_view> rounding;
    switch (mode) {
        case 1:  rounding = "trunc"; break;
        case 2:  rounding = "floor"; break;
        default: rounding = c10::nullopt;
    }
    auto safe_other = at::where(other == 0, at::ones_like(other), other);
    selfRef.div_(safe_other, rounding);
    selfRef.masked_fill_(other == 0, 0);
})

MAKE_OP(aclnnDivMods(const aclTensor* self, const aclScalar* other, int mode, out aclTensor* out,
                     uint64_t* workspaceSize, aclOpExecutor** executor) {
    c10::optional<c10::string_view> rounding;
    switch (mode) {
        case 1:  rounding = "trunc"; break;
        case 2:  rounding = "floor"; break;
        default: rounding = c10::nullopt;
    }
    if (other.equal(0))
        out.zero_();
    else
        at::div_out(out, self, other, rounding);
})
MAKE_OP(aclnnInplaceDivMods(out aclTensor* selfRef, const aclScalar* other, int mode,
                            uint64_t* workspaceSize, aclOpExecutor** executor) {
    c10::optional<c10::string_view> rounding;
    switch (mode) {
        case 1:  rounding = "trunc"; break;
        case 2:  rounding = "floor"; break;
        default: rounding = c10::nullopt;
    }
    if (other.equal(0))
        selfRef.zero_();
    else
        selfRef.div_(other, rounding);
})


MAKE_OP(aclnnPowTensorTensor(const aclTensor* self, const aclTensor* exponent, out aclTensor* out,
                             uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::pow_out(out, self, exponent);
})
MAKE_OP(aclnnInplacePowTensorTensor(out const aclTensor* self, const aclTensor* exponent,
                                    uint64_t* workspaceSize, aclOpExecutor** executor) {
    self.pow_(exponent);
})

MAKE_OP(aclnnPowTensorScalar(const aclTensor* self, const aclScalar* exponent, out aclTensor* out,
                             uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::pow_out(out, self, exponent);
})
MAKE_OP(aclnnInplacePowTensorScalar(out const aclTensor* self, const aclScalar* exponent,
                                    uint64_t* workspaceSize, aclOpExecutor** executor) {
    self.pow_(exponent);
})

MAKE_OP(aclnnPowScalarTensor(const aclScalar* self, const aclTensor* exponent, out aclTensor* out,
                             uint64_t* workspaceSize, aclOpExecutor** executor) {
    out.copy_(at::pow(self, exponent));
})


MAKE_OP(aclnnIsClose(const aclTensor* self, const aclTensor* other, double rtol, double atol, bool equal_nan,
                     out aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto diff = at::abs(self - other);
    auto tol = atol + rtol * at::abs(other);
    auto close = diff <= tol;

    if (equal_nan)
        close |= at::isnan(self) & at::isnan(other);

    out.copy_(close);
})


MAKE_OP(aclnnAtan2(const aclTensor* self, const aclTensor* other, out aclTensor* out,
                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::atan2_out(out, self, other);
})
MAKE_OP(aclnnInplaceAtan2(out aclTensor* selfRef, const aclTensor* other,
                          uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.atan2_(other);
})


MAKE_OP(aclnnXLogYTensor(const aclTensor* self, const aclTensor* other, out aclTensor* out,
                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::xlogy_out(out, self, other);
})
MAKE_OP(aclnnInplaceXLogYTensor(out aclTensor* selfRef, const aclTensor* other,
                                uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.xlogy_(other);
})

MAKE_OP(aclnnXLogYScalarOther(const aclTensor* self, const aclScalar* other, out aclTensor* out,
                              uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::xlogy_out(out, self, other);
})
MAKE_OP(aclnnInplaceXLogYScalarOther(out aclTensor* selfRef, const aclScalar* other,
                                     uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.xlogy_(other);
})

MAKE_OP(aclnnXLogYScalarSelf(const aclScalar* self, const aclTensor* other, out aclTensor* out,
                             uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::xlogy_out(out, self, other);
})

MAKE_OP(aclnnLogAddExp(const aclTensor* self, const aclTensor* other, out aclTensor* out,
                       uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::logaddexp_out(out, self, other);
})
MAKE_OP(aclnnLogAddExp2(const aclTensor* self, const aclTensor* other, out aclTensor* out,
                        uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::logaddexp2_out(out, self, other);
})


MAKE_OP(aclnnFloorDivide(const aclTensor* self, const aclTensor* other, out aclTensor* out,
                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto safe_other = at::where(other == 0, at::ones_like(other), other);
    at::div_out(out, self, safe_other, "floor");
    out.masked_fill_(other == 0, 0);
})
MAKE_OP(aclnnInplaceFloorDivide(out aclTensor* selfRef, const aclTensor* other,
                                uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto safe_other = at::where(other == 0, at::ones_like(other), other);
    selfRef.div_(safe_other, "floor");
    selfRef.masked_fill_(other == 0, 0);
})

MAKE_OP(aclnnFloorDivides(const aclTensor* self, const aclScalar* other, out aclTensor* out,
                          uint64_t* workspaceSize, aclOpExecutor** executor) {
    if (other.equal(0))
        out.zero_();
    else
        at::div_out(out, self, other, "floor");
})
MAKE_OP(aclnnInplaceFloorDivides(out aclTensor* selfRef, const aclScalar* other,
                                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    if (other.equal(0))
        selfRef.zero_();
    else
        selfRef.div_(other, "floor");
})


MAKE_OP(aclnnRemainderTensorTensor(const aclTensor* self, const aclTensor* other, out aclTensor* out,
                                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto safe_other = at::where(other == 0, at::ones_like(other), other);
    at::remainder_out(out, self, safe_other);
})
MAKE_OP(aclnnInplaceRemainderTensorTensor(out aclTensor* selfRef, const aclTensor* other,
                                          uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto safe_other = at::where(other == 0, at::ones_like(other), other);
    selfRef.remainder_(safe_other);
})

MAKE_OP(aclnnRemainderTensorScalar(const aclTensor* self, const aclScalar* other, out aclTensor* out,
                                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    if (other.equal(0))
        out.zero_();
    else
        at::remainder_out(out, self, other);
})
MAKE_OP(aclnnInplaceRemainderTensorScalar(out aclTensor* selfRef, const aclScalar* other,
                                          uint64_t* workspaceSize, aclOpExecutor** executor) {
    if (other.equal(0))
        selfRef.zero_();
    else
        selfRef.remainder_(other);
})

MAKE_OP(aclnnRemainderScalarTensor(const aclScalar* self, const aclTensor* other, out aclTensor* out,
                                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto self_t = at::scalar_tensor(self, other.options());
    auto safe_other = at::where(other == 0, at::ones_like(other), other);
    at::remainder_out(out, self_t, safe_other);
})


MAKE_OP(aclnnAddcdiv(const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2,
                     const aclScalar* value, out aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto safe_t2 = at::where(tensor2 == 0, at::ones_like(tensor2), tensor2);
    at::addcdiv_out(out, self, tensor1, safe_t2, value);
})
MAKE_OP(aclnnInplaceAddcdiv(out aclTensor* selfRef, const aclTensor* tensor1, const aclTensor* tensor2,
                            const aclScalar* value, uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto safe_t2 = at::where(tensor2 == 0, at::ones_like(tensor2), tensor2);
    selfRef.addcdiv_(tensor1, safe_t2, value);
})

MAKE_OP(aclnnAddcmul(const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2,
                     const aclScalar* value, out aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::addcmul_out(out, self, tensor1, tensor2, value);
})
MAKE_OP(aclnnInplaceAddcmul(out aclTensor* selfRef, const aclTensor* tensor1, const aclTensor* tensor2,
                            const aclScalar* value, uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.addcmul_(tensor1, tensor2, value);
})

MAKE_OP(aclnnBatchMatMul(const aclTensor* self, const aclTensor* mat2, out aclTensor* out,
                         int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::matmul_out(out, self, mat2);
})


// ~~~ компараторы ~~~

MAKE_OP(aclnnEqScalar(const aclTensor* self, const aclScalar* other, out aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::eq_out(out, self, other);
})
MAKE_OP(aclnnEqTensor(const aclTensor* self, const aclTensor* other, sync aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    exec->out->store(at::eq(self, other));
})
MAKE_OP(aclnnEqual(const aclTensor* self, const aclTensor* other, sync aclTensor* out,
                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    exec->out->store(at::eq(self, other));
})

MAKE_OP(aclnnNeScalar(const aclTensor* self, const aclScalar* other, out aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::ne_out(out, self, other);
})
MAKE_OP(aclnnNeTensor(const aclTensor* self, const aclTensor* other, sync aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    exec->out->store(at::ne(self, other));
})

MAKE_OP(aclnnLtScalar(const aclTensor* self, const aclScalar* other, out aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::less_out(out, self, other);
})
MAKE_OP(aclnnLtTensor(const aclTensor* self, const aclTensor* other, sync aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    exec->out->store(at::less(self, other));
})

MAKE_OP(aclnnLeScalar(const aclTensor* self, const aclScalar* other, out aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::le_out(out, self, other);
})
MAKE_OP(aclnnLeTensor(const aclTensor* self, const aclTensor* other, sync aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    exec->out->store(at::le(self, other));
})

MAKE_OP(aclnnGtScalar(const aclTensor* self, const aclScalar* other, out aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::greater_out(out, self, other);
})
MAKE_OP(aclnnGtTensor(const aclTensor* self, const aclTensor* other, sync aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    exec->out->store(at::greater(self, other));
})

MAKE_OP(aclnnGeScalar(const aclTensor* self, const aclScalar* other, out aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::ge_out(out, self, other);
})
MAKE_OP(aclnnGeTensor(const aclTensor* self, const aclTensor* other, sync aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    exec->out->store(at::ge(self, other));
})


// ~~~ индексация ~~~

MAKE_OP(aclnnIndex(const aclTensor* self, const aclTensorList* indices, sync aclTensor* out,
                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    // Преобразуем ArrayRef<Tensor> -> List<optional<Tensor>>
    c10::List<std::optional<at::Tensor>> idx_list;
    for (const auto& t : indices)
        idx_list.push_back(t);
    exec->out->store(at::index(self, idx_list));
})
MAKE_OP(aclnnIndexAdd(const aclTensor* self, const int64_t dim, const aclTensor* index,
                      const aclTensor* source, const aclScalar* alpha, sync aclTensor* out,
                      uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::index_add_out(out, self, dim, index, source, alpha);
})
MAKE_OP(aclnnIndexAddV2(const aclTensor* self, const int64_t dim, const aclTensor* index,
                        const aclTensor* source, const aclScalar* alpha, int64_t mode,
                        sync aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    if (mode == 0) {
        at::index_add_out(out, self, dim, index, source, alpha);
    } else {
        out.copy_(self);
        out.index_put_({index}, source * alpha, /*accumulate=*/true);
    }
})

MAKE_OP(aclnnIndexCopy(const aclTensor* selfRef, int64_t dim, const aclTensor* index,
                       const aclTensor* source, out aclTensor* outRef,
                       uint64_t* workspaceSize, aclOpExecutor** executor) {
    outRef.copy_(selfRef);
    outRef.index_copy_(dim, index, source);
})
MAKE_OP(aclnnInplaceIndexCopy(out aclTensor* selfRef, int64_t dim, const aclTensor* index,
                              const aclTensor* source, uint64_t* workspaceSize,
                              aclOpExecutor** executor) {
    selfRef.index_copy_(dim, index, source);
})

MAKE_OP(aclnnIndexFill(const aclTensor* self, int64_t dim, const aclTensor* index, const aclScalar* value,
                       out aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    out.copy_(self);
    out.index_fill_(dim, index, value);
})
MAKE_OP(aclnnInplaceIndexFill(out aclTensor* selfRef, int64_t dim, const aclTensor* index,
                              const aclScalar* value, uint64_t* workspaceSize,
                              aclOpExecutor** executor) {
    selfRef.index_fill_(dim, index, value);
})

MAKE_OP(aclnnIndexFillTensor(const aclTensor* self, int64_t dim, const aclIntArray* index,
                             const aclScalar* value, sync aclTensor* out,
                             uint64_t* workspaceSize, aclOpExecutor** executor) {
    auto index_tensor = at::from_blob(const_cast<int64_t*>(index.data()), index.size(), at::kLong);
    out.copy_(self);
    out.index_fill_(dim, index_tensor, value);
})
MAKE_OP(aclnnInplaceIndexFillTensor(out aclTensor* selfRef, int64_t dim, const aclIntArray* index,
                                    const aclScalar* value, uint64_t* workspaceSize,
                                    aclOpExecutor** executor) {
    auto index_tensor = at::from_blob(const_cast<int64_t*>(index.data()), index.size(), at::kLong);
    selfRef.index_fill_(dim, index_tensor, value);
})

MAKE_OP(aclnnIndexPutImpl(out aclTensor* selfRef, const aclTensorList* indices,
                          const aclTensor* values, const bool accumulate, const bool unsafe,
                          uint64_t* workspaceSize, aclOpExecutor** executor) {
    c10::List<std::optional<at::Tensor>> idx_list;
    for (const auto& t : indices)
        idx_list.push_back(t);
    selfRef.index_put_(idx_list, values, accumulate);
})

MAKE_OP(aclnnIndexSelect(const aclTensor* self, int64_t dim, const aclTensor* index,
                         out aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::index_select_out(out, self, dim, index);
})


MAKE_OP(aclnnSWhere(const aclTensor* condition, const aclTensor* self, const aclTensor* other, out aclTensor* out,
                    uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::where_out(out, condition, self, other);
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
    // at::masked_select_out(out, self, mask);
    exec->out->store(at::masked_select(self, mask));
})


MAKE_OP(aclnnInplaceMaskedFillScalar(out aclTensor* selfRef, const aclTensor* mask,
                                     const aclScalar* value, uint64_t* workspaceSize,
                                     aclOpExecutor** executor) {
    selfRef.masked_fill_(mask, value);
})
MAKE_OP(aclnnInplaceMaskedFillTensor(out aclTensor* selfRef, const aclTensor* mask,
                                     const aclTensor* value, uint64_t* workspaceSize,
                                     aclOpExecutor** executor) {
    selfRef.masked_fill_(mask, value);
})


MAKE_OP(aclnnLogicalAnd(const aclTensor* self, const aclTensor* other, out aclTensor* out,
                        uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::logical_and_out(out, self, other);
})
MAKE_OP(aclnnInplaceLogicalAnd(out aclTensor* selfRef, const aclTensor* other,
                               uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.logical_and_(other);
})

MAKE_OP(aclnnLogicalNot(const aclTensor* self, out aclTensor* out,
                        uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::logical_not_out(out, self);
})
MAKE_OP(aclnnInplaceLogicalNot(out aclTensor* selfRef,
                               uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.logical_not_();
})

MAKE_OP(aclnnLogicalOr(const aclTensor* self, const aclTensor* other, out aclTensor* out,
                       uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::logical_or_out(out, self, other);
})
MAKE_OP(aclnnInplaceLogicalOr(out aclTensor* selfRef, const aclTensor* other,
                              uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.logical_or_(other);
})

MAKE_OP(aclnnLogicalXor(const aclTensor* self, const aclTensor* other, out aclTensor* out,
                        uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::logical_xor_out(out, self, other);
})
MAKE_OP(aclnnInplaceLogicalXor(out aclTensor* selfRef, const aclTensor* other,
                               uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.logical_xor_(other);
})


MAKE_OP(aclnnBitwiseAndScalar(const aclTensor* self, const aclScalar* other, out aclTensor* out,
                              uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::bitwise_and_out(out, self, other);
})
MAKE_OP(aclnnInplaceBitwiseAndScalar(out aclTensor* selfRef, const aclScalar* other,
                                    uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.bitwise_and_(other);
})
MAKE_OP(aclnnBitwiseAndTensor(const aclTensor* self, const aclTensor* other, out aclTensor* out,
                              uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::bitwise_and_out(out, self, other);
})
MAKE_OP(aclnnInplaceBitwiseAndTensor(out aclTensor* selfRef, const aclTensor* other,
                                    uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.bitwise_and_(other);
})
MAKE_OP(aclnnBitwiseAndTensorOut(const aclTensor* self, const aclTensor* other, out aclTensor* out,
                                 uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::bitwise_and_out(out, self, other);
})
MAKE_OP(aclnnInplaceBitwiseAndTensorOut(out aclTensor* selfRef, const aclTensor* other,
                                        uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.bitwise_and_(other);
})

MAKE_OP(aclnnBitwiseNot(const aclTensor* self, out aclTensor* out,
                        uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::bitwise_not_out(out, self);
})
MAKE_OP(aclnnInplaceBitwiseNot(out aclTensor* selfRef,
                              uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.bitwise_not_();
})

MAKE_OP(aclnnBitwiseOrScalar(const aclTensor* self, const aclScalar* other, out aclTensor* out,
                             uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::bitwise_or_out(out, self, other);
})
MAKE_OP(aclnnInplaceBitwiseOrScalar(out aclTensor* selfRef, const aclScalar* other,
                                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.bitwise_or_(other);
})
MAKE_OP(aclnnBitwiseOrTensor(const aclTensor* self, const aclTensor* other, out aclTensor* out,
                             uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::bitwise_or_out(out, self, other);
})
MAKE_OP(aclnnInplaceBitwiseOrTensor(out aclTensor* selfRef, const aclTensor* other,
                                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.bitwise_or_(other);
})

MAKE_OP(aclnnBitwiseXorScalar(const aclTensor* self, const aclScalar* other, out aclTensor* out,
                              uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::bitwise_xor_out(out, self, other);
})
MAKE_OP(aclnnInplaceBitwiseXorScalar(out aclTensor* selfRef, const aclScalar* other,
                                    uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.bitwise_xor_(other);
})
MAKE_OP(aclnnBitwiseXorTensor(const aclTensor* self, const aclTensor* other, out aclTensor* out,
                              uint64_t* workspaceSize, aclOpExecutor** executor) {
    at::bitwise_xor_out(out, self, other);
})
MAKE_OP(aclnnInplaceBitwiseXorTensor(out aclTensor* selfRef, const aclTensor* other,
                                    uint64_t* workspaceSize, aclOpExecutor** executor) {
    selfRef.bitwise_xor_(other);
})


// ~~~ Тяжёлые слои ~~~

MAKE_OP(aclnnFlashAttentionScoreV4(
    const aclTensor* query, const aclTensor* key, const aclTensor* value,
    const aclTensor* realShiftOptional, const aclTensor* dropMaskOptional, const aclTensor* paddingMaskOptional, const aclTensor* attenMaskOptional,
    const aclTensor* queryRopeOptional, const aclTensor* keyRopeOptional, const aclTensor* dScaleQOptional, const aclTensor* dScaleKOptional,
    const aclTensor* dScaleVOptional, const aclTensor* sinkOptional, const aclIntArray* prefixOptional, const aclIntArray* actualSeqQLenOptional,
    const aclIntArray* actualSeqKvLenOptional, const aclIntArray* qStartIdxOptional, const aclIntArray* kvStartIdxOptional,
    double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens, int64_t headNum,
    char* inputLayout, int64_t innerPrecise, int64_t sparseMode, int64_t outDtype, int64_t pseType,
    char* softmaxOutLayout, int64_t seed, int64_t offset,
    sync aclTensor* softmaxMaxOut, sync aclTensor* softmaxSumOut, sync aclTensor* softmaxOutOut, out aclTensor* attentionOutOut,
    uint64_t* workspaceSize, aclOpExecutor** executor
) {
    double scale = scaleValue;
    if (scale == 0.0) scale = 1.0 / std::sqrt(static_cast<double>(query.size(-1)));

    auto attn_weights = at::matmul(query, key.transpose(-2, -1)) * scale;

    // Пытаемся применить маски, только если их форма совместима с attn_weights
    auto try_add_mask = [&](const at::Tensor& mask) {
        if (mask.defined() && mask.numel() > 0) {
            auto broadcast_shape = at::infer_size(attn_weights.sizes(), mask.sizes());
            if (broadcast_shape == attn_weights.sizes() || broadcast_shape == mask.sizes()) {
                attn_weights += mask;
            // иначе просто игнорируем
        }
    };
    try_add_mask(attenMaskOptional);
    try_add_mask(paddingMaskOptional);
    try_add_mask(realShiftOptional);

    auto attn_probs = at::softmax(attn_weights, -1);
    auto result = at::matmul(attn_probs, value);

    attentionOutOut.copy_(result);

    if (softmaxMaxOut.defined() && softmaxMaxOut.numel() > 0)
        exec->softmaxMaxOut->store(std::get<0>(at::max(attn_weights, -1, true)));
    if (softmaxSumOut.defined() && softmaxSumOut.numel() > 0)
        exec->softmaxSumOut->store(at::sum(attn_probs, -1, true));
    if (softmaxOutOut.defined() && softmaxOutOut.numel() > 0)
        exec->softmaxOutOut->store(attn_probs);
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

    if (!log_is_quiet()) {
        log << '\n'; tensorDataToString(tensor, log);
        log_output(log);
    }

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
    auto tensor = at::from_blob(value, {}, opts).clone();
    return new aclScalar {
        .item = tensor.item(),
        .tensor = tensor,
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
        << "\nError: UNIMPLEMENTED BaseOp";
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclGetStorageShape(const aclTensor* tensor, int64_t** storageDims, uint64_t* storageDimsNum) {
    std::ostringstream log;
    log << "[aclGetStorageShape] tensor=" << static_cast<const void*>(tensor)
        << " storageDims=" << static_cast<void*>(storageDims)
        << " storageDimsNum=" << static_cast<void*>(storageDimsNum)
        << "\nError: UNIMPLEMENTED BaseOp";
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclGetViewStrides(const aclTensor* tensor, int64_t** stridesValue, uint64_t* stridesNum) {
    std::ostringstream log;
    log << "[aclGetViewStrides] tensor=" << static_cast<const void*>(tensor)
        << " stridesValue=" << static_cast<void*>(stridesValue)
        << " stridesNum=" << static_cast<void*>(stridesNum)
        << "\nError: UNIMPLEMENTED BaseOp";
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclGetViewOffset(const aclTensor* tensor, int64_t* offset) {
    std::ostringstream log;
    log << "[aclGetViewOffset] tensor=" << static_cast<const void*>(tensor)
        << " offset=" << static_cast<void*>(offset)
        << "\nError: UNIMPLEMENTED BaseOp";
    log_output(log, true);
    return UNIMPLEMENTED;
}

aclnnStatus aclGetFormat(const aclTensor* tensor, aclFormat* format) {
    std::ostringstream log;
    log << "[aclGetFormat] tensor=" << static_cast<const void*>(tensor)
        << " format=" << static_cast<void*>(format)
        << "\nError: UNIMPLEMENTED BaseOp";
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
        << "\nError: UNIMPLEMENTED BaseOp";
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
// weak не работает, если функции определены в одном и том же cpp-скрипте -_-

DEFINE_UNIMPLEMENTED_ACLNN(aclStftGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* windowOptional, aclTensor* out, int64_t nFft, int64_t hopLength,
                           int64_t winLength, bool normalized, bool onesided, bool returnComplex, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclStft,
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

DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddbmmGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* batch1, const aclTensor* batch2, const aclScalar* beta,
                           const aclScalar* alpha, aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnAddbmm,
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

DEFINE_UNIMPLEMENTED_ACLNN(aclnnBincountGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* weights, int64_t minlength, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnBincount,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)

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

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGatherNdGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* indices,
                           bool negativeIndexSupport, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGatherNd,
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

DEFINE_UNIMPLEMENTED_ACLNN(aclnnGeluQuantGetWorkspaceSize,
                           const aclTensor* self, const aclTensor* inputScaleOptional, const aclTensor* inputOffsetOptional,
                           const char* approximate, const char* quantMode, const char* roundMode, int64_t dstType, const aclTensor* y,
                           const aclTensor* outScaleOptional, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnGeluQuant,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

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

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAddReluGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* other, aclScalar* alpha,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAddRelu,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAddbmmGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* batch1, const aclTensor* batch2, const aclScalar* beta, const aclScalar* alpha,
                           int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceAddbmm,
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

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceCumprodGetWorkspaceSize,
                           aclTensor *input, const aclScalar *dim, uint64_t *workspaceSize, aclOpExecutor **executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceCumprod,
                           void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                           aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceEqScalarGetWorkspaceSize,
                           const aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceEqScalar,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceEqTensorGetWorkspaceSize,
                           const aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceEqTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceFfnWorkerSchedulerGetWorkspaceSize,
                           aclTensor* scheduleContextRef, int32_t syncGroupSize,
                           int32_t executeMode, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceFfnWorkerScheduler,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

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

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLeScalarGetWorkspaceSize,
                           aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLeScalar,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLeTensorGetWorkspaceSize,
                           aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLeTensor,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLtScalarGetWorkspaceSize,
                           const aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLtScalar,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLtTensorGetWorkspaceSize,
                           const aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceLtTensor,
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

DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceRenormGetWorkspaceSize,
                           aclTensor* selfRef, const aclScalar* p, int64_t dim, const aclScalar* maxNorm, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnInplaceRenorm,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

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

DEFINE_UNIMPLEMENTED_ACLNN(aclnnNonMaxSuppressionGetWorkspaceSize,
                           const aclTensor* boxes, const aclTensor* scores,
                           aclIntArray* maxOutputBoxesPerClass,
                           aclFloatArray* iouThreshold, aclFloatArray* scoreThreshold,
                           int32_t centerPointBox, aclTensor* selectedIndices,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnNonMaxSuppression,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                           aclrtStream stream)

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

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRealGetWorkspaceSize,
                           const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnReal,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

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

DEFINE_UNIMPLEMENTED_ACLNN(aclnnRenormGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* p, int64_t dim, const aclScalar* maxNorm, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnRenorm,
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

DEFINE_UNIMPLEMENTED_ACLNN(aclnnShrinkGetWorkspaceSize,
                           const aclTensor* self, const aclScalar* lambd, const aclScalar* bias, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
DEFINE_UNIMPLEMENTED_ACLNN(aclnnShrink,
                           void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)

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


#ifdef __cplusplus
}
#endif
