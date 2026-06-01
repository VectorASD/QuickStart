# базовая логика:

LIB_BASE="$HOME/tmp/Ascend-cann/run_package"
LIB_BASE_OPS="$HOME/tmp/Ascend-cann-950/run_package"
SYMBOL=aclAppLog

LIBS=(
    "$LIB_BASE/cann-npu-runtime/runtime/lib/libascendcl.so"
    "$LIB_BASE/cann-npu-runtime/runtime/lib/libruntime.so"
    "$LIB_BASE/cann-npu-runtime/runtime/lib/libprofapi.so"
    "$LIB_BASE/cann-ge-compiler/ge-compiler/lib64/libacl_op_compiler.so"
    "$LIB_BASE_OPS/cann-hccl/hccl/lib64/libhccl.so"
)

function cut_base() {
    local short="$1"
    case "$short" in
        "$LIB_BASE"/*)
            short="${short#$LIB_BASE/}"
            ;;
        "$LIB_BASE_OPS"/*)
            short="(ops) ${short#$LIB_BASE_OPS/}"
            ;;
    esac
    echo "$short"
}

function symbol_finder() {
    local SYMBOL="$1"
    for f in ${LIBS[*]}; do
        nm -D "$f" 2>/dev/null | grep -F $SYMBOL && echo " → $(cut_base $f)"
    done
}

function lib_finder() {
    # общий поиск либы и заголовков по символу
    local SYMBOL="$1"
    for f in $(find $LIB_BASE -name '*.so*') $(find $LIB_BASE_OPS -name '*.so*'); do
        nm -D "$f" 2>/dev/null | grep -F $SYMBOL && echo " → $f"
    done
    grep -Rnw $LIB_BASE -e $SYMBOL --include='*.h' 2>/dev/null
}


# точечный поиск по либам и заголовкам:

for f in ${LIBS[*]}; do
    nm -D "$f" 2>/dev/null | grep -F $SYMBOL && echo " → $(cut_base $f)"
done

INCLUDES=(
    cann-npu-runtime/runtime/include/external/acl/acl_base_rt.h
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
    cann-npu-runtime/runtime/pkg_inc/profiling/aprof_pub.h
    cann-npu-runtime/runtime/pkg_inc/runtime/runtime/kernel.h
    cann-ge-executor/ge-executor/include/acl/acl_base_mdl.h
    cann-ge-executor/ge-executor/include/acl/acl_op.h
    cann-ge-compiler/ge-compiler/include/acl/acl_op_compiler.h
)
for f in ${INCLUDES[*]}; do
    grep -F "$SYMBOL(" "$LIB_BASE/$f" 2>/dev/null && echo " → $f"
done


# сборщик символов

SYM_BASE=$(for f in $(find $LIB_BASE -name '*.so*'); do
    nm -D --defined-only "$f" 2>/dev/null \
        | awk '$3 !~ /_/ && $3 !~ /@/ && $3 ~ /[A-Z]/ {print $3}'
done | sort -u)
echo "$SYM_BASE"

TORCH_NPU_LIBS=/opt/python311/lib/python3.11/site-packages/torch_npu/lib
#   статическая линковка:
# REQ=$(for f in $(find $TORCH_NPU_LIBS -name '*.so*'); do
#     nm -D --undefined-only "$f" 2>/dev/null \
#         | awk '$NF !~ /_/ && $NF !~ /@/ && $NF ~ /[A-Z]/ {print $NF}'
# done | sort -u)
# echo "$REQ"
#   динамическая загрузка функций из torch_npu (собраны посредством register_finder.py):
REQ=(AmlAicoreDetectOnline AmlP2PDetectOnline HcclAllGather HcclAllGatherV HcclAllReduce HcclAlltoAll HcclAlltoAllV HcclBatchSendRecv HcclBroadcast HcclCommActivateCommMemory HcclCommDeactivateCommMemory HcclCommDeregister HcclCommDestroy HcclCommExchangeMem HcclCommInitAll HcclCommInitClusterInfoConfig HcclCommInitRootInfo HcclCommInitRootInfoConfig HcclCommRegister HcclCommResume HcclCommSetMemoryRange HcclCommUnsetMemoryRange HcclCommWorkingDevNicSet HcclCreateSubCommConfig HcclGetCommAsyncError HcclGetCommConfigCapability HcclGetCommName HcclGetRootInfo HcclGroupEnd HcclGroupStart HcclRecv HcclReduce HcclReduceScatter HcclReduceScatterV HcclScatter HcclSend HcclSetConfig LcalCommInit LcalCommInitRankLocal LcclAllGather LcclAllReduce LcclBroadcast LcclCommDestroy LcclReduceScatter aclCreateGraphDumpOpt aclDestroyAclOpExecutor aclDestroyGraphDumpOpt aclGenGraphAndDumpForOp aclGetCannAttribute aclGetCannAttributeList aclGetCompileopt aclGetCompileoptSize aclGetDeviceCapability aclGetRecentErrMsg aclSetCompileopt aclmdlRICaptureBegin aclmdlRICaptureEnd aclmdlRICaptureGetInfo aclmdlRICaptureTaskGrpBegin aclmdlRICaptureTaskGrpEnd aclmdlRICaptureTaskUpdateBegin aclmdlRICaptureTaskUpdateEnd aclmdlRICaptureThreadExchangeMode aclmdlRIDebugJsonPrint aclmdlRIDebugPrint aclmdlRIDestroy aclmdlRIExecuteAsync aclnnReselectStaticKernel aclnnSilentCheck aclnnSilentCheckV2 aclopCompileAndExecuteV2 aclopStartDumpArgs aclopStopDumpArgs aclprofCreateConfig aclprofCreateStepInfo aclprofDestroyConfig aclprofDestroyStepInfo aclprofFinalize aclprofGetStepTimestamp aclprofGetSupportedFeatures aclprofGetSupportedFeaturesV2 aclprofInit aclprofMarkEx aclprofRegisterDeviceCallback aclprofSetConfig aclprofStart aclprofStop aclprofWarmup aclrtCmoAsync aclrtCreateEventExWithFlag aclrtCreateEventWithFlag aclrtCreateStream aclrtCreateStreamWithConfig aclrtCtxSetSysParamOpt aclrtDestroyStreamForce aclrtDeviceCanAccessPeer aclrtDeviceGetBareTgid aclrtDeviceGetUuid aclrtDeviceTaskAbort aclrtEventGetTimestamp aclrtFreePhysical aclrtGetDeviceInfo aclrtGetDeviceResLimit aclrtGetDeviceUtilizationRate aclrtGetErrorVerbose aclrtGetLastError aclrtGetMemUceInfo aclrtGetMemUsageInfo aclrtGetPrimaryCtxState aclrtGetResInCurrentThread aclrtGetSocName aclrtGetStreamOverflowSwitch aclrtGetStreamResLimit aclrtHostRegister aclrtHostRegisterV2 aclrtHostUnregister aclrtIpcGetEventHandle aclrtIpcMemClose aclrtIpcMemGetExportKey aclrtIpcMemImportByKey aclrtIpcMemSetImportPid aclrtIpcOpenEventHandle aclrtLaunchCallback aclrtLaunchHostFunc aclrtMallocAlign32 aclrtMallocHostWithCfg aclrtMallocPhysical aclrtMapMem aclrtMemExportToShareableHandle aclrtMemImportFromShareableHandle aclrtMemSetPidToShareableHandle aclrtMemUceRepair aclrtMemcpyAsyncWithCondition aclrtMemcpyBatch aclrtMemcpyBatchAsync aclrtMemset aclrtPeekAtLastError aclrtPointerGetAttributes aclrtQueryEventStatus aclrtQueryEventWaitStatus aclrtReleaseMemAddress aclrtRepairError aclrtReserveMemAddress aclrtResetDeviceResLimit aclrtResetStreamResLimit aclrtSetDeviceResLimit aclrtSetDeviceSatMode aclrtSetOpExecuteTimeOut aclrtSetOpExecuteTimeOutV2 aclrtSetOpWaitTimeout aclrtSetStreamAttribute aclrtSetStreamFailureMode aclrtSetStreamOverflowSwitch aclrtSetStreamResLimit aclrtSetSysParamOpt aclrtStreamGetId aclrtStreamQuery aclrtSubscribeReport aclrtSynchronizeDevice aclrtSynchronizeDeviceWithTimeout aclrtSynchronizeStream aclrtSynchronizeStreamWithTimeout aclrtUnSubscribeReport aclrtUnmapMem aclrtUnuseStreamResInCurrentThread aclrtUseStreamResInCurrentThread aclrtValueWait aclrtValueWrite aclshmem_finalize aclshmem_free aclshmem_malloc aclshmem_ptr aclshmemx_get_uniqueid aclshmemx_getmem_on_stream aclshmemx_init_attr aclshmemx_putmem_on_stream aclshmemx_set_attr_uniqueid_args aclshmemx_set_conf_store_tls aclskOptimize aclskScopeBegin aclskScopeEnd aclsysGetCANNVersion aclsysGetVersionStr dcmi_get_affinity_cpu_info_by_device_id dcmi_get_card_num_list dcmi_get_device_id_in_card dcmi_init dcmiv2_get_affinity_cpu_info_by_dev_id dcmiv2_get_affinity_cpu_info_by_device_id dcmiv2_get_device_list dcmiv2_init halGetAPIVersion halGetDeviceInfo mstxDomainCreateA mstxDomainDestroy mstxDomainMarkA mstxDomainRangeEnd mstxDomainRangeStartA mstxMarkA mstxMemHeapRegister mstxMemHeapUnregister mstxMemRegionsRegister mstxMemRegionsUnregister mstxRangeEnd mstxRangeStartA shmem_finalize shmem_free shmem_get_uniqueid shmem_malloc shmem_ptr shmem_set_attr shmem_set_attr_uniqueid_args shmem_set_conf_store_tls)
REQ="${REQ[*]}"

for sym in $REQ; do
    echo "$sym"
    for lib in ${LIBS[*]}; do
        if nm -D --defined-only "$lib" 2>/dev/null | grep -qw "$sym"; then
            echo "    $(cut_base $lib)"
        fi
    done
    for f in ${INCLUDES[*]}; do
        if grep -q "$sym(" "$LIB_BASE/$f" 2>/dev/null; then
            echo "    $f"
        fi
    done
    IFS=':' read -ra DIRS <<< "$LD_LIBRARY_PATH"
    unset available;
    for d in "${DIRS[@]}"; do
        for so in "$d"/*.so*; do
            [ -e "$so" ] || continue
            if nm -D --defined-only "$so" 2>/dev/null | grep -qw "$sym"; then
              # echo "    [ld.so] available via $so"
                available=''
            fi
        done
    done
    if [ ! -v available ]; then
        echo "    UNRELEASE"
    fi
done


: << 'COMMENT'
# финальный результат анализа символов

MsprofGetHashId
    cann-npu-runtime/runtime/lib/libprofapi.so
    cann-npu-runtime/runtime/pkg_inc/profiling/aprof_pub.h
MsprofReportApi
    cann-npu-runtime/runtime/lib/libprofapi.so
    cann-npu-runtime/runtime/pkg_inc/profiling/aprof_pub.h
MsprofReportCompactInfo
    cann-npu-runtime/runtime/lib/libprofapi.so
    cann-npu-runtime/runtime/pkg_inc/profiling/aprof_pub.h
MsprofSysCycleTime
    cann-npu-runtime/runtime/lib/libprofapi.so
    cann-npu-runtime/runtime/pkg_inc/profiling/aprof_pub.h
ParameterClass
THPDeviceType   (здесь все *Type и *Class относятся к pytorch)
THPDtypeType
THPEventClass
THPStorageClass
THPStreamClass
THPVariableClass
aclAppLog
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_base_rt.h
aclCreateDataBuffer
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_base_rt.h
aclCreateTensorDesc
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_base_mdl.h
aclDestroyDataBuffer
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_base_rt.h
aclDestroyTensorDesc
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_base_mdl.h
aclFinalize
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclGetDeviceCapability
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_base_rt.h
aclGetRecentErrMsg
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclGetTensorDescDimV2
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_base_mdl.h
aclGetTensorDescFormat
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_base_mdl.h
aclGetTensorDescNumDims
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_base_mdl.h
aclGetTensorDescType
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_base_mdl.h
aclInit
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclSetTensorDescName
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_base_mdl.h
aclSetTensorFormat
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_base_mdl.h
aclSetTensorPlaceMent
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_base_mdl.h
aclSetTensorShape
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_base_mdl.h
aclmdlFinalizeDump
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclmdlInitDump
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclmdlSetDump
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclopCompileAndExecute
    cann-ge-compiler/ge-compiler/lib64/libacl_op_compiler.so
    cann-ge-compiler/ge-compiler/include/acl/acl_op_compiler.h
aclopCreateAttr
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_op.h
aclopDestroyAttr
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_op.h
aclopSetAttrBool
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_op.h
aclopSetAttrDataType
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_op.h
aclopSetAttrFloat
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_op.h
aclopSetAttrInt
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_op.h
aclopSetAttrListBool
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_op.h
aclopSetAttrListFloat
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_op.h
aclopSetAttrListInt
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_op.h
aclopSetAttrListListInt
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_op.h
aclopSetAttrString
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-ge-executor/ge-executor/include/acl/acl_op.h
aclrtCreateStream
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtDestroyEvent
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtDestroyStream
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtDeviceCanAccessPeer
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtDeviceEnablePeerAccess
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtEventElapsedTime
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtFree
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtFreeHost
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtGetCurrentContext
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtGetDevice
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtGetDeviceCount
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtGetDeviceInfo
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtGetMemInfo
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtMalloc
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtMallocHost
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtMemcpy
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtMemcpyAsync
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtMemset
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtProcessReport
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtRecordEvent
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtResetDevice
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtResetEvent
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtSetCurrentContext
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtSetDevice
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtStreamWaitEvent
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtSynchronizeEvent
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
aclrtSynchronizeStream
    cann-npu-runtime/runtime/lib/libascendcl.so
    cann-npu-runtime/runtime/include/external/acl/acl_rt.h
rtDevBinaryRegister
    cann-npu-runtime/runtime/lib/libruntime.so
    cann-npu-runtime/runtime/pkg_inc/runtime/runtime/kernel.h
rtFunctionRegister
    cann-npu-runtime/runtime/lib/libruntime.so
    cann-npu-runtime/runtime/pkg_inc/runtime/runtime/kernel.h
rtKernelLaunch
    cann-npu-runtime/runtime/lib/libruntime.so
    cann-npu-runtime/runtime/pkg_inc/runtime/runtime/kernel.h
COMMENT
