from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import re

def merge_continued_lines(lines: List[str]) -> List[str]:
    """Склеивает строки, заканчивающиеся на \\, с последующей строкой."""
    merged = []
    current = ""
    for line in lines:
        stripped = line.rstrip('\n')
        if stripped.endswith('\\'):
            current += stripped[:-1]  # убираем \ в конце
        else:
            current += stripped
            merged.append(current)
            current = ""
    if current:  # если последняя строка заканчивалась на \\
        merged.append(current)
    return merged

def find_register_calls(root_dir: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Собирает все регистрации функций через TORCH_NPU_REGISTER_FUNCTION (прямые и через обёртки)
    и возвращает словарь {имя_библиотеки: [список_функций]}.

    Параметры:
        root_dir: корневая директория (по умолчанию ~/tmp/pytorch/torch_npu/csrc)
    """
    if root_dir is None:
        root_dir = Path.home() / "tmp" / "pytorch" / "torch_npu" / "csrc"

    extensions = {'.cpp', '.c', '.cc', '.cxx', '.h', '.hpp', '.hxx'}
    lib_functions: Dict[str, List[str]] = defaultdict(list)

    # Паттерны
    # 1. Определение оборачивающего макроса: #define ИМЯ(ПАРАМ) ... TORCH_NPU_REGISTER_FUNCTION(ЛИБА, ПАРАМ)
    define_wrapper_re = re.compile(
        r'#define\s+(\w+)\((\w+)\)\s+(.*)'  # захватываем имя макроса, параметр и хвост
    )
    # В хвосте ищем TORCH_NPU_REGISTER_FUNCTION(библиотека, параметр)
    register_in_define_re = re.compile(
        r'TORCH_NPU_REGISTER_FUNCTION\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)'
    )

    # 2. Вызов известного макроса: ИМЯ(функция)
    macro_call_re = re.compile(r'\b(\w+)\s*\(\s*(\w+)\s*\)')

    # 3. Прямой вызов TORCH_NPU_REGISTER_FUNCTION(либа, функция)
    direct_register_re = re.compile(
        r'TORCH_NPU_REGISTER_FUNCTION\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)'
    )

    for file_path in root_dir.rglob('*'):
        if file_path.suffix not in extensions:
            continue
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_lines = f.readlines()
        except Exception:
            continue

        logical_lines = merge_continued_lines(raw_lines)
        active_wrappers: Dict[str, str] = {}  # имя макроса -> библиотека

        for line in logical_lines:
            stripped = line.lstrip()
            # --- Обработка препроцессорных директив ---
            if stripped.startswith('#define'):
                m = define_wrapper_re.match(stripped)
                if m:
                    macro_name, param, rest = m.groups()
                    # Ищем TORCH_NPU_REGISTER_FUNCTION в остатке строки
                    reg_m = register_in_define_re.search(rest)
                    if reg_m:
                        lib, used_param = reg_m.groups()
                        # убедимся, что используется именно тот же параметр
                        if used_param == param:
                            active_wrappers[macro_name] = lib
                # Если это определение других макросов, просто игнорируем
                continue

            elif stripped.startswith('#undef'):
                parts = stripped.split()
                if len(parts) >= 2:
                    macro_name = parts[1]
                    active_wrappers.pop(macro_name, None)
                continue

            # --- Обычный код ---
            # 1. Вызовы оборачивающих макросов
            for call_m in macro_call_re.finditer(line):
                name, arg = call_m.groups()
                if name in active_wrappers:
                    lib = active_wrappers[name]
                    lib_functions[lib].append(arg)

            # 2. Прямые вызовы TORCH_NPU_REGISTER_FUNCTION
            for direct_m in direct_register_re.finditer(line):
                lib, func = direct_m.groups()
                lib_functions[lib].append(func)

    # Удаляем дубликаты, сохраняя порядок появления
    result = {}
    for lib, funcs in lib_functions.items():
        seen = set()
        unique_funcs = []
        for f in funcs:
            if f not in seen:
                seen.add(f)
                unique_funcs.append(f)
        result[lib] = unique_funcs
    return result

def get_implemented_functions(impl_dir: Path) -> Dict[str, List[str]]:
    """
    Сканирует все .cpp/.c файлы в impl_dir и строит словарь:
        имя_функции -> список файлов, в которых встречается вызов/определение этой функции.
    """
    impl_files: Dict[str, str] = {}
    for ext in ('*.cpp', '*.c'):
        for file_path in impl_dir.glob(ext):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    impl_files[file_path.name] = f.read()
            except Exception:
                continue

    func_locations: Dict[str, List[str]] = defaultdict(list)
    func_pattern = re.compile(r'\b([a-zA-Z_]\w*)\s*\(')

    for fname, content in impl_files.items():
        for match in func_pattern.finditer(content):
            func_name = match.group(1)
            func_locations[func_name].append(fname)

    # удаляем дубликаты файлов, сохраняя порядок первого появления
    result = {}
    for func, files in func_locations.items():
        seen = set()
        unique_files = []
        for f in files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)
        result[func] = unique_files
    return result


if __name__ == "__main__":
    npu_root = Path.home() / "tmp" / "pytorch" / "torch_npu" / "csrc"
    impl_dir = Path(__file__).parent / "src"

    registrations = find_register_calls(npu_root)
    if not registrations:
        print("Регистрации не найдены.")
        exit()

    impl_map = get_implemented_functions(impl_dir)
    print("!!!", impl_map)

    print("Собранные регистрации по библиотекам и статусам реализации:")
    for lib in sorted(registrations):
        print(f"- {lib}:")
        for func in sorted(registrations[lib]):
            files = impl_map.get(func, [])
            if not files:
                files.append('x')
            status = ", ".join(files)
            print(f"    {func}: ({status})")


"""
Собранные регистрации по библиотекам и статусам реализации:
- libacl_op_compiler:
    aclCreateGraphDumpOpt: (x)
    aclDestroyGraphDumpOpt: (x)
    aclGenGraphAndDumpForOp: (x)
    aclGetCompileopt: (x)
    aclGetCompileoptSize: (x)
    aclSetCompileopt: (not_acl_op_compiler.cpp)
    aclopCompileAndExecuteV2: (x)
    aclrtCtxSetSysParamOpt: (not_acl.cpp)
    aclrtSetSysParamOpt: (x)
- libascend_dump:
    aclopStartDumpArgs: (x)
    aclopStopDumpArgs: (x)
- libascend_hal:
    halGetAPIVersion: (x)
    halGetDeviceInfo: (x)
- libascend_ml:
    AmlAicoreDetectOnline: (x)
    AmlP2PDetectOnline: (x)
- libascendcl:
    aclGetCannAttribute: (x)
    aclGetCannAttributeList: (x)
    aclGetDeviceCapability: (not_acl.cpp)
    aclGetRecentErrMsg: (not_acl.cpp)
    aclmdlRICaptureBegin: (x)
    aclmdlRICaptureEnd: (x)
    aclmdlRICaptureGetInfo: (x)
    aclmdlRICaptureTaskGrpBegin: (x)
    aclmdlRICaptureTaskGrpEnd: (x)
    aclmdlRICaptureTaskUpdateBegin: (x)
    aclmdlRICaptureTaskUpdateEnd: (x)
    aclmdlRICaptureThreadExchangeMode: (x)
    aclmdlRIDebugJsonPrint: (x)
    aclmdlRIDebugPrint: (x)
    aclmdlRIDestroy: (x)
    aclmdlRIExecuteAsync: (x)
    aclprofCreateConfig: (x)
    aclprofCreateStepInfo: (x)
    aclprofDestroyConfig: (x)
    aclprofDestroyStepInfo: (x)
    aclprofFinalize: (x)
    aclprofGetStepTimestamp: (x)
    aclprofInit: (x)
    aclprofStart: (x)
    aclprofStop: (x)
    aclrtCmoAsync: (x)
    aclrtCreateEventExWithFlag: (x)
    aclrtCreateEventWithFlag: (x)
    aclrtCreateStream: (not_acl.cpp)
    aclrtCreateStreamWithConfig: (x)
    aclrtDestroyStreamForce: (x)
    aclrtDeviceCanAccessPeer: (not_acl.cpp)
    aclrtDeviceGetBareTgid: (x)
    aclrtDeviceGetUuid: (x)
    aclrtDeviceTaskAbort: (x)
    aclrtEventGetTimestamp: (x)
    aclrtFreePhysical: (x)
    aclrtGetDeviceInfo: (not_acl.cpp)
    aclrtGetDeviceResLimit: (x)
    aclrtGetDeviceUtilizationRate: (x)
    aclrtGetErrorVerbose: (x)
    aclrtGetLastError: (x)
    aclrtGetMemUceInfo: (x)
    aclrtGetMemUsageInfo: (x)
    aclrtGetPrimaryCtxState: (x)
    aclrtGetResInCurrentThread: (x)
    aclrtGetSocName: (x)
    aclrtGetStreamOverflowSwitch: (x)
    aclrtGetStreamResLimit: (x)
    aclrtHostRegister: (x)
    aclrtHostRegisterV2: (x)
    aclrtHostUnregister: (x)
    aclrtIpcGetEventHandle: (x)
    aclrtIpcMemClose: (x)
    aclrtIpcMemGetExportKey: (x)
    aclrtIpcMemImportByKey: (x)
    aclrtIpcMemSetImportPid: (x)
    aclrtIpcOpenEventHandle: (x)
    aclrtLaunchCallback: (x)
    aclrtLaunchHostFunc: (x)
    aclrtMallocAlign32: (x)
    aclrtMallocHostWithCfg: (x)
    aclrtMallocPhysical: (x)
    aclrtMapMem: (x)
    aclrtMemExportToShareableHandle: (x)
    aclrtMemImportFromShareableHandle: (x)
    aclrtMemSetPidToShareableHandle: (x)
    aclrtMemUceRepair: (x)
    aclrtMemcpyAsyncWithCondition: (x)
    aclrtMemcpyBatch: (x)
    aclrtMemcpyBatchAsync: (x)
    aclrtMemset: (not_acl.cpp)
    aclrtPeekAtLastError: (x)
    aclrtPointerGetAttributes: (x)
    aclrtQueryEventStatus: (x)
    aclrtQueryEventWaitStatus: (x)
    aclrtReleaseMemAddress: (x)
    aclrtRepairError: (x)
    aclrtReserveMemAddress: (x)
    aclrtResetDeviceResLimit: (x)
    aclrtResetStreamResLimit: (x)
    aclrtSetDeviceResLimit: (x)
    aclrtSetDeviceSatMode: (not_acl.cpp)
    aclrtSetOpExecuteTimeOut: (not_acl.cpp)
    aclrtSetOpExecuteTimeOutV2: (x)
    aclrtSetOpWaitTimeout: (x)
    aclrtSetStreamAttribute: (x)
    aclrtSetStreamFailureMode: (x)
    aclrtSetStreamOverflowSwitch: (x)
    aclrtSetStreamResLimit: (x)
    aclrtStreamGetId: (x)
    aclrtStreamQuery: (x)
    aclrtSubscribeReport: (x)
    aclrtSynchronizeDevice: (x)
    aclrtSynchronizeDeviceWithTimeout: (x)
    aclrtSynchronizeStream: (not_acl.cpp)
    aclrtSynchronizeStreamWithTimeout: (x)
    aclrtUnSubscribeReport: (x)
    aclrtUnmapMem: (x)
    aclrtUnuseStreamResInCurrentThread: (x)
    aclrtUseStreamResInCurrentThread: (x)
    aclrtValueWait: (x)
    aclrtValueWrite: (x)
    aclsysGetCANNVersion: (x)
    aclsysGetVersionStr: (x)
- libascendsk:
    aclskOptimize: (x)
    aclskScopeBegin: (x)
    aclskScopeEnd: (x)
- libdcmi:
    dcmi_get_affinity_cpu_info_by_device_id: (x)
    dcmi_get_card_num_list: (x)
    dcmi_get_device_id_in_card: (x)
    dcmi_init: (x)
    dcmiv2_get_affinity_cpu_info_by_dev_id: (x)
    dcmiv2_get_affinity_cpu_info_by_device_id: (x)
    dcmiv2_get_device_list: (x)
    dcmiv2_init: (x)
- libhccl:
    HcclAllGather: (x)
    HcclAllGatherV: (x)
    HcclAllReduce: (x)
    HcclAlltoAll: (x)
    HcclAlltoAllV: (x)
    HcclBatchSendRecv: (x)
    HcclBroadcast: (x)
    HcclCommActivateCommMemory: (x)
    HcclCommDeactivateCommMemory: (x)
    HcclCommDeregister: (x)
    HcclCommDestroy: (x)
    HcclCommExchangeMem: (x)
    HcclCommInitAll: (x)
    HcclCommInitClusterInfoConfig: (x)
    HcclCommInitRootInfo: (x)
    HcclCommInitRootInfoConfig: (x)
    HcclCommRegister: (x)
    HcclCommResume: (x)
    HcclCommSetMemoryRange: (x)
    HcclCommUnsetMemoryRange: (x)
    HcclCommWorkingDevNicSet: (x)
    HcclCreateSubCommConfig: (x)
    HcclGetCommAsyncError: (x)
    HcclGetCommConfigCapability: (x)
    HcclGetCommName: (x)
    HcclGetRootInfo: (x)
    HcclRecv: (x)
    HcclReduce: (x)
    HcclReduceScatter: (x)
    HcclReduceScatterV: (x)
    HcclScatter: (x)
    HcclSend: (x)
    HcclSetConfig: (x)
- libhcomm:
    HcclGroupEnd: (x)
    HcclGroupStart: (x)
- liblcal:
    LcalCommInit: (x)
    LcalCommInitRankLocal: (x)
    LcclAllGather: (x)
    LcclAllReduce: (x)
    LcclBroadcast: (x)
    LcclCommDestroy: (x)
    LcclReduceScatter: (x)
- libms_tools_ext:
    mstxDomainCreateA: (x)
    mstxDomainDestroy: (x)
    mstxDomainMarkA: (x)
    mstxDomainRangeEnd: (x)
    mstxDomainRangeStartA: (x)
    mstxMarkA: (x)
    mstxMemHeapRegister: (x)
    mstxMemHeapUnregister: (x)
    mstxMemRegionsRegister: (x)
    mstxMemRegionsUnregister: (x)
    mstxRangeEnd: (x)
    mstxRangeStartA: (x)
- libmsprofiler:
    aclprofGetSupportedFeatures: (x)
    aclprofGetSupportedFeaturesV2: (x)
    aclprofMarkEx: (x)
    aclprofRegisterDeviceCallback: (x)
    aclprofSetConfig: (x)
    aclprofWarmup: (x)
- libnnopbase:
    aclDestroyAclOpExecutor: (x)
- libopapi:
    aclnnReselectStaticKernel: (x)
    aclnnSilentCheck: (x)
    aclnnSilentCheckV2: (x)
- libshmem:
    aclshmem_finalize: (x)
    aclshmem_free: (x)
    aclshmem_malloc: (x)
    aclshmem_ptr: (x)
    aclshmemx_get_uniqueid: (x)
    aclshmemx_getmem_on_stream: (x)
    aclshmemx_init_attr: (x)
    aclshmemx_putmem_on_stream: (x)
    aclshmemx_set_attr_uniqueid_args: (x)
    aclshmemx_set_conf_store_tls: (x)
    shmem_finalize: (x)
    shmem_free: (x)
    shmem_get_uniqueid: (x)
    shmem_malloc: (x)
    shmem_ptr: (x)
    shmem_set_attr: (x)
    shmem_set_attr_uniqueid_args: (x)
    shmem_set_conf_store_tls: (x)
"""
