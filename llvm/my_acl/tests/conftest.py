import pytest

import ctypes
import gc
import os
import pty
from pathlib import Path
import psutil
import re
import select
import signal
import stat
import subprocess
import sys
import threading



# ---------- memory check ----------
libc = ctypes.CDLL("libc.so.6")

def check_memory(threshold: float = 0.6):
    vm = psutil.virtual_memory()
    usage = (vm.total - vm.available) / vm.total  # vm.percent / 100
    # Killed после 0.777* :) Не могу найти формулу, чтобы Killed был около 1
  # print(usage, vm.percent)
    if usage > threshold:
        gc.collect()
        libc.malloc_trim(0)
        vm = psutil.virtual_memory()
        usage2 = (vm.total - vm.available) / vm.total
        if usage2 > threshold * 0.9:
            print(f"USED MEMORY: {usage * 100:.3f}% -> {usage2 * 100:.3f}%")
            raise RuntimeError("Memory leak detected! Freed less than 10% of used memory")

@pytest.fixture(autouse=True)
def memory_check(request):
    check_memory()
    yield
    check_memory()

@pytest.fixture(autouse=True)
def check_fallback(capfd):
    yield
    captured = capfd.readouterr()
    if "CAUTION" in captured.err:
        ops = []
        for line in captured.err.splitlines():
            match = re.search(r"operator '([^']+)'", line)
            if match:
                ops.append(match.group(1))
        ops_list = "\n".join(f"  - {op}" for op in ops) if ops else "unknown"
        pytest.fail(f"NPU fallback to CPU detected!\nUnsported operations:\n{ops_list}\n\nOriginal stderr:\n{captured.err}")



# ---------- Attrs ----------
class Attrs:
    __slots__ = ('bold', 'italic', 'underline', 'strike', 'reverse', 'fg', 'bg')
    def __init__(self, bold=False, italic=False, underline=False, strike=False,
                 reverse=False, fg=None, bg=None):
        self.bold = bold
        self.italic = italic
        self.underline = underline
        self.strike = strike
        self.reverse = reverse
        self.fg = fg
        self.bg = bg

    def copy(self):
        return Attrs(self.bold, self.italic, self.underline, self.strike,
                     self.reverse, self.fg, self.bg)

    def __eq__(self, other):
        if not isinstance(other, Attrs):
            return False
        return (self.bold == other.bold and self.italic == other.italic and
                self.underline == other.underline and self.strike == other.strike and
                self.reverse == other.reverse and self.fg == other.fg and self.bg == other.bg)

DEFAULT_ATTRS = Attrs()

def _has_visible_effect(attrs: Attrs) -> bool:
    """True, если атрибуты делают пробельный символ видимым."""
    return attrs.underline or attrs.strike or attrs.reverse or (attrs.bg is not None)

def diff_attrs(old: Attrs, new: Attrs) -> str:
    if old == new:
        return ''
    if new == DEFAULT_ATTRS:
        return '\\e[0m'
    reset_codes, set_codes = [], []
    # Сбросы
    if old.bold and not new.bold: reset_codes.append(22)
    if old.italic and not new.italic: reset_codes.append(23)
    if old.underline and not new.underline: reset_codes.append(24)
    if old.strike and not new.strike: reset_codes.append(29)
    if old.reverse and not new.reverse: reset_codes.append(27)
    # Цвета: сбрасываем только когда переходим к None, иначе просто установим новый
    if old.fg is not None and new.fg is None:
        reset_codes.append(39)
    if old.bg is not None and new.bg is None:
        reset_codes.append(49)
    # Установки
    if new.bold and not old.bold: set_codes.append(1)
    if new.italic and not old.italic: set_codes.append(3)
    if new.underline and not old.underline: set_codes.append(4)
    if new.strike and not old.strike: set_codes.append(9)
    if new.reverse and not old.reverse: set_codes.append(7)
    if new.fg is not None and (old.fg is None or old.fg != new.fg):
        set_codes.append(new.fg)
    if new.bg is not None and (old.bg is None or old.bg != new.bg):
        set_codes.append(new.bg)
    codes = reset_codes + set_codes
    return '\\e[' + ';'.join(map(str, codes)) + 'm' if codes else ''

# ---------- VirtualTerminal ----------
class VirtualTerminal:
    def __init__(self, log_path, width=80, max_height=10000):
        self.width = width
        self.max_height = max_height
        self.current_attrs = DEFAULT_ATTRS.copy()
        self.last_written_attrs = DEFAULT_ATTRS.copy()
        self.cursor_row = 0
        self.cursor_col = 0

        # отслеживание самой ранней изменённой ячейки с момента последней синхронизации
        self.dirty_start_row = None
        self.dirty_start_col = None

        log_path.touch(exist_ok=True)
        log_path.chmod(log_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.fd = log_path.open('w+b')

        magic = b'body=$(tail -n +2 "$0"); printf "%b\\n" "$body"; exit\n'
        self.fd.write(magic)
        self.magic_len = len(magic)
        self.file_start = self.fd.tell()

        # экран: список строк, каждая строка – список [char, attrs, offset]
        self.screen = []
        # смещения начала каждой строки в файле
        self.line_offsets = []

    # -----------------------------------------------------------------
    #  Публичный интерфейс
    # -----------------------------------------------------------------
    def feed(self, text: str):
        i = 0
        n = len(text)
        while i < n:
            c = text[i]
            if c == '\x1b':                     # escape-последовательность
                i += 1
                if i >= n: break
                next_c = text[i]
                if next_c == '[':               # CSI
                    i += 1
                    params_str = ''
                    while i < n and text[i] not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
                        params_str += text[i]
                        i += 1
                    if i >= n: break
                    command = text[i]
                    i += 1
                    self._csi_dispatch(command, params_str)
                else:
                    i += 1   # игнорируем другие escape
            elif c == '\n':
                # перед переходом на новую строку – финализируем текущую, если она изменялась
                if self.dirty_start_row is not None:
                    self._flush_dirty_region()
                    self.dirty_start_row = None
                    self.dirty_start_col = None
                # записываем физический перевод строки в лог
                self.fd.write(b'\n')
                self._linefeed()
                # запоминаем начало новой строки
                if len(self.line_offsets) <= self.cursor_row:
                    self.line_offsets.extend([None] * (self.cursor_row - len(self.line_offsets) + 1))
                self.line_offsets[self.cursor_row] = self.fd.tell()
                # НЕ сбрасываем last_written_attrs – атрибуты наследуются с предыдущей строки
                i += 1
            elif c == '\r':
                self.cursor_col = 0
                i += 1
            elif c == '\b':
                if self.cursor_col > 0:
                    self.cursor_col -= 1
                i += 1
            elif c == '\t':
                tabsize = 8
                self.cursor_col = ((self.cursor_col // tabsize) + 1) * tabsize
                if self.cursor_col >= self.width:
                    self.cursor_col = self.width - 1
                i += 1
            elif c == '\x07':                   # BEL
                i += 1
            else:
                self._put_char(c)
                i += 1

        # в конце ввода финализируем последнюю изменённую строку
        if self.dirty_start_row is not None:
            self._flush_dirty_region()
            self.dirty_start_row = None
            self.dirty_start_col = None
            self.fd.flush()

    def close(self):
        # Если последние атрибуты не дефолтные, сбрасываем их в лог
        if self.last_written_attrs != DEFAULT_ATTRS:
            diff = diff_attrs(self.last_written_attrs, DEFAULT_ATTRS)
            if diff:
                self._write_raw(diff)
        self.fd.close()

    # -----------------------------------------------------------------
    #  Основные функции: put и flush
    # -----------------------------------------------------------------
    def _update_dirty(self, row: int, col: int):
        """Помечает ячейку (row, col) как минимальную границу грязного региона."""
        if self.dirty_start_row is None:
            self.dirty_start_row = row
            self.dirty_start_col = col
        elif row < self.dirty_start_row or \
             (row == self.dirty_start_row and col < self.dirty_start_col):
            self.dirty_start_row = row
            self.dirty_start_col = col

    def _put_char(self, char: str):
        """Добавляет/заменяет символ в позиции курсора и двигает курсор."""
        # autowrap
        if self.cursor_col >= self.width:
            self.cursor_col = 0
            self.cursor_row += 1
            if self.cursor_row >= self.max_height:
                self._scroll_up()

        self._update_dirty(self.cursor_row, self.cursor_col)

        # гарантируем существование строки и столбца в матрице
        while len(self.screen) <= self.cursor_row:
            self.screen.append([])
        line = self.screen[self.cursor_row]
        while len(line) <= self.cursor_col:
            line.append([' ', DEFAULT_ATTRS.copy(), None])

        cell = line[self.cursor_col]
        cell[0] = char
        cell[1] = self.current_attrs.copy()
        cell[2] = None
        self.cursor_col += 1

    def _linefeed(self):
        """Обрабатывает перевод строки (\n)."""
        self.cursor_row += 1
        if self.cursor_row >= self.max_height:
            self._scroll_up()
        # гарантируем, что новая строка существует в матрице
        while len(self.screen) <= self.cursor_row:
            self.screen.append([])

    def _scroll_up(self):
        """Сдвигает экран вверх на одну строку."""
        if len(self.screen) > 0:
            self.screen.pop(0)
            # также удаляем соответствующий line_offset
            if self.line_offsets:
                self.line_offsets.pop(0)
        self.screen.append([])
        self.cursor_row = self.max_height - 1

    def _flush_dirty_region(self):
        if self.dirty_start_row is None:
            return

        row = self.dirty_start_row
        col = self.dirty_start_col
        if row >= len(self.screen):
            return
        line = self.screen[row]

        # 1. Определяем позицию в файле и начальные атрибуты
        if col == 0:
            # Начало строки: ищем позицию начала строки в файле
            if row < len(self.line_offsets) and self.line_offsets[row] is not None:
                self.fd.seek(self.line_offsets[row])
            else:
                self.fd.seek(0, os.SEEK_END)
                if row >= len(self.line_offsets):
                    self.line_offsets.extend([None] * (row - len(self.line_offsets) + 1))
                self.line_offsets[row] = self.fd.tell()
            # Атрибуты перед началом строки — от последнего значащего символа выше
            last_attrs = self._get_effective_attrs(row - 1, self.width - 1)
        else:
            # Ищем последнюю записанную в файл ячейку строго левее col
            pos = None
            prev_attrs = None
            for c in range(col - 1, -1, -1):
                if c < len(line) and line[c][2] is not None:
                    pos = line[c][2] + len(line[c][0].encode('utf-8'))
                    prev_attrs = line[c][1].copy()
                    break
            if pos is not None:
                self.fd.seek(pos)
                last_attrs = prev_attrs
            else:
                # Записанных ячеек левее нет — начинаем с начала строки
                if row < len(self.line_offsets) and self.line_offsets[row] is not None:
                    self.fd.seek(self.line_offsets[row])
                else:
                    self.fd.seek(0, os.SEEK_END)
                    if row >= len(self.line_offsets):
                        self.line_offsets.extend([None] * (row - len(self.line_offsets) + 1))
                    self.line_offsets[row] = self.fd.tell()
                last_attrs = self._get_effective_attrs(row, col - 1)

        # 2. Определяем последний значащий столбец в строке
        last_sig = -1
        for c in range(len(line)):
            ch, attrs, _ = line[c]
            if ch != ' ' or attrs != DEFAULT_ATTRS:
                last_sig = c

        # 3. Особый случай: грязь находится правее всех значащих символов
        if col > last_sig:
            if last_sig >= 0:
                last_cell = line[last_sig]
                if last_cell[2] is not None:
                    self.fd.seek(last_cell[2] + len(last_cell[0].encode('utf-8')))
                # Проверяем, изменились ли атрибуты для будущих символов
                current_eff = self._get_effective_attrs(row, col - 1)
                if current_eff != last_attrs:
                    diff = diff_attrs(last_attrs, current_eff)
                    if diff:
                        self._write_raw(diff)
                        last_attrs = current_eff.copy()
            # Усекаем хвост (до следующей строки)
            if row + 1 < len(self.line_offsets) and self.line_offsets[row+1] is not None:
                next_start = self.line_offsets[row+1]
                if self.fd.tell() < next_start:
                    self.fd.truncate(next_start)
            else:
                self.fd.truncate()
            self.last_written_attrs = last_attrs
            return

        # 4. Обычная перезапись от col до last_sig
        while col <= last_sig:
            if col < len(line):
                cell = line[col]
            else:
                cell = [' ', DEFAULT_ATTRS.copy(), None]
            char = cell[0]
            attrs = cell[1]

            if attrs != last_attrs:
                if char == ' ' and not _has_visible_effect(attrs):
                    self._write_char(' ', last_attrs)
                else:
                    diff = diff_attrs(last_attrs, attrs)
                    if diff:
                        self._write_raw(diff)
                        last_attrs = attrs.copy()
                    self._write_char(char, last_attrs)
            else:
                self._write_char(char, last_attrs)

            while len(line) <= col:
                line.append([' ', DEFAULT_ATTRS.copy(), None])
            line[col][0] = char
            line[col][1] = attrs.copy()
            line[col][2] = self.fd.tell() - len(char.encode('utf-8'))
            col += 1

        # 5. Обрезаем хвост
        if row + 1 < len(self.line_offsets) and self.line_offsets[row+1] is not None:
            next_start = self.line_offsets[row+1]
            if self.fd.tell() < next_start:
                self.fd.truncate(next_start)
        else:
            self.fd.truncate()

        self.last_written_attrs = last_attrs

    # -----------------------------------------------------------------
    #  Низкоуровневый вывод
    # -----------------------------------------------------------------
    def _write_raw(self, data: str):
        self.fd.write(data.encode('utf-8'))

    def _write_char(self, char: str, attrs: Attrs = None):
        if attrs is None:
            attrs = self.last_written_attrs
        if char == '\\':
            self.fd.write(b'\\\\')
        else:
            self.fd.write(char.encode('utf-8'))

    # -----------------------------------------------------------------
    #  Обработка CSI-команд
    # -----------------------------------------------------------------
    def _get_effective_attrs(self, row: int, col: int) -> Attrs:
        """
        Ищет ближайший значащий символ слева от (row, col) или на предыдущих строках.
        Значащим считается символ, который НЕ является пробелом без видимых атрибутов
        (т.е. если пробел подчёркнут/зачёркнут/инвертирован или имеет фон – он видим).
        """
        r, c = row, col
        while r >= 0:
            if r >= len(self.screen):
                r -= 1
                c = self.width - 1
                continue
            line = self.screen[r]
            while c >= 0:
                if c < len(line):
                    ch, attrs, _ = line[c]
                    # игнорируем только пробелы без видимых эффектов
                    if ch != ' ' or _has_visible_effect(attrs):
                        return attrs.copy()
                c -= 1
            r -= 1
            c = self.width - 1
        return DEFAULT_ATTRS.copy()

    def _csi_dispatch(self, command: str, params_str: str):
        params = []
        if params_str.strip():
            for p in params_str.split(';'):
                if p == '' or p == '?':
                    params.append(0)
                else:
                    try: params.append(int(p))
                    except ValueError: params.append(0)
        else:
            params = [0]

        if command == 'm':
            self._sgr(params)
            if self.cursor_row < len(self.screen):
                self._update_dirty(self.cursor_row, self.cursor_col)
        elif command == 'A':  # вверх
            n = params[0] if params else 1
            self.cursor_row = max(0, self.cursor_row - n)
        elif command == 'B':  # вниз
            n = params[0] if params else 1
            self.cursor_row = min(self.max_height - 1, self.cursor_row + n)
        elif command == 'C':  # вправо
            n = params[0] if params else 1
            self.cursor_col = min(self.width - 1, self.cursor_col + n)
        elif command == 'D':  # влево
            n = params[0] if params else 1
            self.cursor_col = max(0, self.cursor_col - n)
        elif command == 'E':  # на следующую строку, в начало
            n = params[0] if params else 1
            self.cursor_row = min(self.max_height - 1, self.cursor_row + n)
            self.cursor_col = 0
        elif command == 'F':  # на предыдущую строку, в начало
            n = params[0] if params else 1
            self.cursor_row = max(0, self.cursor_row - n)
            self.cursor_col = 0
        elif command == 'G':  # абсолютная горизонтальная позиция
            n = params[0] if params else 1
            self.cursor_col = max(0, min(self.width - 1, n - 1))
        elif command in 'Hf':  # абсолютная позиция (H или f) – двигаем курсор, в лог не пишем
            row = params[0] - 1 if len(params) > 0 else 0
            col = params[1] - 1 if len(params) > 1 else 0
            self.cursor_row = max(0, min(self.max_height - 1, row))
            self.cursor_col = max(0, min(self.width - 1, col))
        elif command == 'J':  # очистка дисплея
            mode = params[0] if params else 0
            self._erase_display(mode)
        elif command == 'K':  # очистка строки
            mode = params[0] if params else 0
            self._erase_line(mode)

    def _sgr(self, codes):
        i = 0
        while i < len(codes):
            c = codes[i]
            if c == 0: self.current_attrs = DEFAULT_ATTRS.copy()
            elif c == 1: self.current_attrs.bold = True
            elif c == 3: self.current_attrs.italic = True
            elif c == 4: self.current_attrs.underline = True
            elif c == 7: self.current_attrs.reverse = True
            elif c == 9: self.current_attrs.strike = True
            elif c == 22: self.current_attrs.bold = False
            elif c == 23: self.current_attrs.italic = False
            elif c == 24: self.current_attrs.underline = False
            elif c == 27: self.current_attrs.reverse = False
            elif c == 29: self.current_attrs.strike = False
            elif 30 <= c <= 37: self.current_attrs.fg = c
            elif c == 39: self.current_attrs.fg = None
            elif 40 <= c <= 47: self.current_attrs.bg = c
            elif c == 49: self.current_attrs.bg = None
            elif 90 <= c <= 97: self.current_attrs.fg = c
            elif 100 <= c <= 107: self.current_attrs.bg = c
            i += 1

    # -----------------------------------------------------------------
    #  Очистка строк / экрана (влияет только на матрицу, не на файл)
    # -----------------------------------------------------------------
    def _erase_line(self, mode: int):
        if self.cursor_row >= len(self.screen): return
        if mode == 0:
            self._erase_line_in_row(self.cursor_row, self.cursor_col, self.width)
        elif mode == 1:
            self._erase_line_in_row(self.cursor_row, 0, self.cursor_col)
        elif mode == 2:
            self._erase_line_in_row(self.cursor_row, 0, self.width)

    def _erase_line_in_row(self, row: int, start_col: int, end_col: int):
        if row >= len(self.screen): return
        line = self.screen[row]
        for col in range(start_col, end_col):
            if col >= self.width: break
            while len(line) <= col:
                line.append([' ', DEFAULT_ATTRS.copy(), None])
            line[col][0] = ' '
            line[col][1] = DEFAULT_ATTRS.copy()
            line[col][2] = None

    def _erase_display(self, mode: int):
        if mode == 0:
            self._erase_line(0)
            for r in range(self.cursor_row + 1, len(self.screen)):
                self._erase_line_in_row(r, 0, self.width)
        elif mode == 1:
            self._erase_line(1)
            for r in range(0, self.cursor_row):
                self._erase_line_in_row(r, 0, self.width)
        elif mode == 2:
            self.screen = []
            self.cursor_row = 0
            self.cursor_col = 0

# ---------- Subprocess with PTY ----------
def get_terminal_size():
    try:
        import termios, fcntl, struct
        with open('/dev/tty', 'rb') as tty:
            fd = tty.fileno()
            winsize = struct.pack('HHHH', 0, 0, 0, 0)
            winsize = fcntl.ioctl(fd, termios.TIOCGWINSZ, winsize)
            rows, cols, _, _ = struct.unpack('HHHH', winsize)
            if rows > 0 and cols > 0:
                return rows, cols
    except Exception: pass
    try:
        import shutil
        cols, rows = shutil.get_terminal_size((80, 24))
        return rows, cols
    except Exception: return 24, 80

def set_teminal_size(slave_fd, cols, rows):
    try:
        import termios, fcntl, struct
        winsize = struct.pack('HHHH', rows, cols, 0, 0)
        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)
    except: pass

def is_child_process():
    try:
        current = psutil.Process()
        parent = current.parent()
        if parent is None: return False
        return current.exe() == parent.exe()
    except: return False

def run_pytest_via_subprocess(log_path):
    env = os.environ.copy()
    rows, cols = get_terminal_size()
    args = [sys.executable, '-m', 'pytest'] + sys.argv[1:]

    master_fd, slave_fd = pty.openpty()
    set_teminal_size(slave_fd, cols, rows)

    proc = subprocess.Popen(
        args,
        stdin=subprocess.DEVNULL,
        stdout=slave_fd,
        stderr=slave_fd,
        env=env,
        pass_fds=(slave_fd,),
        preexec_fn=os.setsid
    )
    os.close(slave_fd)

    terminal = VirtualTerminal(log_path, width=cols, max_height=1000)

    def read_and_process():
        nonlocal proc
        while proc.poll() is None:
            rlist, _, _ = select.select([master_fd], [], [], 0.1)
            if master_fd in rlist:
                try:
                    data = os.read(master_fd, 4096)
                    if not data: break
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
                    text = data.decode('utf-8', errors='replace')
                    terminal.feed(text)
                except OSError: break
        # Дочитываем остатки
        while True:
            try:
                data = os.read(master_fd, 4096)
                if not data: break
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
                terminal.feed(data.decode('utf-8', errors='replace'))
            except OSError: break
        terminal.close()

    thread = threading.Thread(target=read_and_process, daemon=True)
    thread.start()
    try:
        proc.wait()
    except KeyboardInterrupt:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
    thread.join(timeout=1)
  # sys.exit(proc.returncode) # Тогда после Ctrl+C или любого неудавшегося теста, pytest начнёт много писать "INTERNALERROR> ..."
    os._exit(proc.returncode)



def make_log_path(config):
    test_files = [arg for arg in config.args if arg.endswith('.py')]
    assert len(test_files) == 1, test_files
    test_file = Path(test_files[0]).resolve()
    log_dir = test_file.parent / "log"
    log_dir.mkdir(exist_ok=True)
    stem = test_file.stem
    if stem.startswith("test_"):
        stem = stem[len("test_"):]
    return log_dir / f"{stem}.log"

def pytest_addoption(parser):
    parser.addoption("--log", action="store_true", default=False, help="Enable virtual terminal logging")

def pytest_configure(config):
    if config.getoption("log", default=False):
        if is_child_process():
            config.option.capture = "fd"
        else:
            config.option.capture = "no"
            log_path = make_log_path(config)
            run_pytest_via_subprocess(log_path)
