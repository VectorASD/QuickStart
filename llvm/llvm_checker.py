from pathlib import Path
from pprint import pprint
import os
import sys

RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RESET  = "\033[0m"

prefix = sys.argv[2] if len(sys.argv) > 2 else '/usr/lib/llvm-20/bin'

ref = {
  'Backtrace_LIBRARY':              '',
  'CMAKE_ADDR2LINE':                    f'{prefix}/llvm-symbolizer', # llvm-addr2line
  'CMAKE_AR':                           f'{prefix}/llvm-ar',
  'CMAKE_ASM_COMPILER':                 f'{prefix}/clang-23',
  'CMAKE_ASM_COMPILER_AR':              f'{prefix}/llvm-ar',
  'CMAKE_ASM_COMPILER_CLANG_SCAN_DEPS': f'{prefix}/clang-scan-deps',
  'CMAKE_ASM_COMPILER_RANLIB':          f'{prefix}/llvm-ar', # llvm-ranlib
  'CMAKE_CXX_COMPILER_AR':              f'{prefix}/llvm-ar',
  'CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS': f'{prefix}/clang-scan-deps',
  'CMAKE_CXX_COMPILER_RANLIB':          f'{prefix}/llvm-ar', # llvm-ranlib
  'CMAKE_C_COMPILER_AR':                f'{prefix}/llvm-ar',
  'CMAKE_C_COMPILER_CLANG_SCAN_DEPS':   f'{prefix}/clang-scan-deps',
  'CMAKE_C_COMPILER_RANLIB':            f'{prefix}/llvm-ar', # llvm-ranlib
  'CMAKE_DLLTOOL':                      f'{prefix}/llvm-ar', # llvm-dlltool
  'CMAKE_LINKER':                       f'{prefix}/lld', # ld.lld
  'CMAKE_MAKE_PROGRAM':                 '/opt/ninja/bin/ninja',
  'CMAKE_NM':                           f'{prefix}/llvm-nm',
  'CMAKE_OBJCOPY':                      f'{prefix}/llvm-objcopy',
  'CMAKE_OBJDUMP':                      f'{prefix}/llvm-objdump',
  'CMAKE_RANLIB':                       f'{prefix}/llvm-ar', # llvm-ranlib
  'CMAKE_READELF':                      f'{prefix}/llvm-readobj', # llvm-readelf
  'CMAKE_STRIP':                        f'{prefix}/llvm-objcopy', # llvm-strip
  'CMAKE_TAPI':                     'CMAKE_TAPI-NOTFOUND',
  'GIT_EXECUTABLE':                     '/usr/bin/git',
  'GOLD_EXECUTABLE':                    '/usr/bin/x86_64-linux-gnu-ld.gold',
  'LLVM_FILECHECK_EXE':                 'LLVM_FILECHECK_EXE-NOTFOUND',
  'LLVM_LOCAL_RPATH':               '',
  'LLVM_PROFDATA_FILE':             '',
  'LLVM_SPROFDATA_FILE':            '',
  'LibEdit_LIBRARIES':              'LibEdit_LIBRARIES-NOTFOUND',
  'OCAMLFIND':                      'OCAMLFIND-NOTFOUND',
  'PKG_CONFIG_EXECUTABLE':              '/usr/bin/pkg-config',
  'ZLIB_LIBRARY_DEBUG':             'ZLIB_LIBRARY_DEBUG-NOTFOUND',
  'ZLIB_LIBRARY_RELEASE':               '/usr/lib/x86_64-linux-gnu/libz.so.1.2.11',
  'zstd_LIBRARY':                   'zstd_LIBRARY-NOTFOUND',
  'zstd_STATIC_LIBRARY':            'zstd_STATIC_LIBRARY-NOTFOUND'
}

# path = Path.home() / "tmp" / "llvm-project" / "build" / "CMakeCache.txt"
path = Path(sys.argv[1] if len(sys.argv) > 1 else "CMakeCache.txt")
base = {}
errors = 0
with path.open() as file:
    for line in file:
        if ":FILEPATH=" in line:
            key, value = line.split(":FILEPATH=")
            value = value.rstrip()
            if value.startswith('/'):
                value = os.path.realpath(value)
            base[key] = value

            ref_value = ref.get(key, '')
            ok    = ref_value.startswith('/')
            error = ok and os.path.realpath(ref_value) != value
            color  = RED if error else GREEN
            color2 = "" if ok or error else YELLOW
            print(f"{color}{key:36} {color2}{value}{RESET}")
            if error:
                print(f"    {YELLOW}expected: {RESET}{ref_value}")
                errors += 1

# pprint(base)

print()
if errors:
    a, b = errors % 100 // 10, errors % 10
    part  = 'ок' if a == 1 or b == 0 or b > 4 else 'ка' if b == 1 else 'ки'
    part2 = 'о'  if a == 1 or b == 0 or b > 4 else  'а' if b == 1 else  'ы'
    print(f"{RED}Обнаружен{part2} {errors} ошиб{part}!{RESET}")
    exit(1)
else:
    print(f"{GREEN}Ошибок нет! ;'-}}{RESET}")
