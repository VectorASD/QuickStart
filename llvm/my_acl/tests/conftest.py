import pytest
import psutil, gc, ctypes



libc = ctypes.CDLL("libc.so.6")

def check_memory(threshold: bool = 0.8):
    vm = psutil.virtual_memory()
    free = vm.used / vm.available
    if free > threshold:
        gc.collect()         # free unused python objects
        libc.malloc_trim(0)  # free gcc allocator CACHE
        vm = psutil.virtual_memory()
        free2 = vm.used / vm.available
      # print(f"USED MEMORY: {free * 100:.3f}% -> {free2 * 100:.3f}%")
        if free2 > threshold * 0.9:
            print(f"USED MEMORY: {free * 100:.3f}% -> {free2 * 100:.3f}%")
            raise RuntimeError("Memory leak detected! Freed less than 10% of used memory")
    """
malloc_trim — функция именно glibc, а не gcc/clang. Пока вы остаётесь на Linux с glibc
(что по умолчанию для большинства дистрибутивов, а в нашем случае, это обычно Ubuntu 22.04),
она будет работать одинаково хорошо и при использовании clang. Компилятор здесь
не важен – важен рантайм C (glibc).
Так что можете смело переходить на clang, проблем с управлением памятью не возникнет.
    """

@pytest.fixture(autouse=True)
def memory_check(request):
    check_memory()  # перед тестом
    yield
    check_memory()  # после теста
