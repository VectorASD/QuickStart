#include "not_opapi_base.h"

struct Executor0 {
    const aclTensor* selfRef;
    const float mean;
    const float std;
    const int64_t seed;
    const int64_t offset;

    void start(const char* opName, std::ostringstream& log) const {
        log << "[EXEC] " << opName << ": mean=" << formatFloat(mean) << " std=" << formatFloat(std) << " seed=" << seed << " offset=" << offset;
    }
    void end(std::ostringstream& log) const {
        log << "\nselfRef:\n" << tensorDataToString(selfRef);
    }
};

struct UnaryExecutor {
    const aclTensor* x;
    const aclTensor* out;

    void start(const char* opName, std::ostringstream& log) const {
        log << "[EXEC] " << opName << ':'
            << "\nx:\n" << tensorDataToString(x);
    }
    void end(std::ostringstream& log) const {
        log << "\nout:\n" << tensorDataToString(out);
    }
};


__attribute__((visibility("default")))
aclnnStatus aclnnInplaceNormalGetWorkspaceSize(const aclTensor* selfRef, const float mean, const float std, const int64_t seed, const int64_t offset, uint64_t* workspaceSize, aclOpExecutor** executor) {
    ASSERT_CODE(workspaceSize && executor, INVALID_PARAM)
    ASSERT(selfRef)
    Executor0* exec = new Executor0{selfRef, mean, std, seed, offset};
    *workspaceSize = 0;
    *executor = reinterpret_cast<aclOpExecutor*>(exec);
    return OK;
}
DEFINE_ACLNN_OP(aclnnInplaceNormal, Executor0, {
    at::Tensor selfRef;
    LOAD_TENSOR(selfRef, exec->selfRef);

    auto gen = at::detail::getDefaultCPUGenerator();
    gen.set_current_seed(static_cast<uint64_t>(exec->seed) + static_cast<uint64_t>(exec->offset));
    selfRef.normal_(exec->mean, exec->std, gen);
})


__attribute__((visibility("default")))
aclnnStatus aclnnAngleV2GetWorkspaceSize(const aclTensor* x, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    ASSERT_CODE(workspaceSize && executor, INVALID_PARAM)
    ASSERT(x && out)
    UnaryExecutor* exec = new UnaryExecutor{x, out};
    *workspaceSize = 0;
    *executor = reinterpret_cast<aclOpExecutor*>(exec);
    return OK;
}
DEFINE_ACLNN_OP(aclnnAngleV2, UnaryExecutor, {
    at::Tensor x, out;
    LOAD_TENSOR(x, exec->x);
    LOAD_TENSOR(out, exec->out);

    at::angle_out(out, x);
})
