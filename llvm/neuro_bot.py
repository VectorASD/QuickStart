import llama_cpp
import os
from pathlib import Path

def update(path = Path.home() / "FlagGems" / "neuro_bot.py"):
    exec(path.read_text(), globals())
    # супер удобный способ прямо из интерактивного python загрузить весь этот скрипт целиком неограниченное количество раз
    # гарантируется, что вы не сможете инициализировать сразу две llm за раз из-за общего globals()

if "llm" not in globals():
    n_threads = os.cpu_count()   # 20 на работе, дома 12, при тех же 16 GB RAM и настройках WSL... из этой же инструкции :) дома чувствуются тормоза, а на работе летает
    llm = llama_cpp.Llama(
        (Path.home() / 'tmp' / 'DeepSeek-V2-Lite.Q4_K_M.gguf').as_posix(),
        n_ctx          = 8192,
        n_threads      = n_threads,
    )

# temperature    = 0.7
# top_p          = 0.95,
# repeat_penalty = 1.1,
# system_prompt  = "You are a helpful English-speaking assistant. Answer user messages directly and concisely. Do not invent dialogue roles. Do not continue articles or websites. Respond only as the assistant.",
# stream         = True,
#   там стоит **kwargs, который делает видимость, что эти параметры куда-то ушли,
#   а на практике это тупо заглушка для обратной совместимости!

def cdiv(div, rem):
    return (div + rem - 1) // rem

def calculate_list_of_bit_sizes():
    GGML_TYPE_BITS = {
        "F32": 32, "F16": 16,
        "Q4_0": 4, "Q4_1": 4, "Q5_0": 5, "Q5_1": 5, "Q8_0": 8, "Q8_1": 8,
        "Q2_K": 2, "Q3_K": 3, "Q4_K": 4, "Q5_K": 5, "Q6_K": 6, "Q8_K": 8,
        "IQ2_XXS": 2, "IQ2_XS": 2, "IQ3_XXS": 3, "IQ1_S": 1, "IQ4_NL": 4, "IQ3_S": 3, "IQ2_S": 2, "IQ4_XS": 4,
        "I8": 8, "I16": 16, "I32": 32, "I64": 64, "F64": 64,
        "IQ1_M": 1, "MXFP4": 4, "NVFP4": 4
    }
    LIST_OF_BITS = [None] * llama_cpp.GGML_TYPE_COUNT
    for k, v in llama_cpp.__dict__.items():
        if k.startswith("GGML_TYPE_") and k != "GGML_TYPE_COUNT":
            LIST_OF_BITS[v] = GGML_TYPE_BITS[k[len("GGML_TYPE_"):]]
    return LIST_OF_BITS

def get_KV_cache_size():
    n_ctx    = llm.input_ids.shape[0]   # llm.context_params.n_ctx
    metadata = llm._model.metadata()

    n_layer   = int(metadata["deepseek2.block_count"])
    n_kv_head = int(metadata["deepseek2.attention.head_count_kv"])
    key_len   = int(metadata["deepseek2.attention.key_length"])
    value_len = int(metadata["deepseek2.attention.value_length"])

    LIST_OF_BITS = calculate_list_of_bit_sizes()
    K_size = cdiv(n_layer * n_ctx * n_kv_head * key_len   * LIST_OF_BITS[llm.context_params.type_k], 8)
    V_size = cdiv(n_layer * n_ctx * n_kv_head * value_len * LIST_OF_BITS[llm.context_params.type_v], 8)

    print(f"K-cache size: {K_size / 1024 ** 3:.3f} Gb.")   # 1.266 Gb.
    print(f"V-cache size: {V_size / 1024 ** 3:.3f} Gb.")   # 0.844 Gb.
    print(f"TOTAL-cache size: {(K_size + V_size) / 1024 ** 3:.3f} Gb.")   # 2.109 Gb.



class EndDetector:
    def __init__(self, word):
        self.tokens = llm.tokenize(word.encode("utf-8"), add_bos=False)

    def match(self, history):
        tokens = self.tokens
        size = len(tokens)
        for i in range(size, 0, -1):
            if history[-i:] == tokens[:i]:
                return i, i == size
        return 0, False

    def complete(self, history):
        count, _ = self.match(history)
        history.extend(self.tokens[count:])

class Utf8Chainer:
    def __init__(self):
        self.buffer = bytearray()
        self.need = 0

    def __call__(self, bytes):
        result = bytearray()
        buffer = self.buffer

        for byte in bytes:
            if self.need:
                buffer.append(byte)
                if byte >> 6 == 0b10: # 10xxxxxx
                    self.need -= 1
                    if not self.need:
                        result += buffer
                        buffer.clear()
                else: # error
                    self.need = 0
                    result += buffer
                    buffer.clear()
            elif byte >> 7 == 0b0: # 0xxxxxxx
                result.append(byte)
            elif byte >> 5 == 0b110: # 110xxxxx
                buffer.append(byte)
                self.need = 1
            elif byte >> 4 == 0b1110: # 1110xxxx
                buffer.append(byte)
                self.need = 2
            elif byte >> 3 == 0b11110: # 11110xxx
                buffer.append(byte)
                self.need = 3
            else: # error
                result.append(byte)
        return result.decode("utf-8", errors="replace")

system_prompt = llm.tokenize(("""Вы - диалоговый ассистент.
Отвечай только после слова "Assistant:"
Отвечай максимально естественно, как человек.
Никогда не пиши "\\nAssistant:".
Не повторяй за пользователем слово в слово.
""").encode("utf-8"), add_bos=True)
assert len(system_prompt) == 79

if "history" not in globals():
    history = []

def core(message="", reset=False):
    if reset:
        llm.reset()
        history.clear()
    if not message:
        return

    ender     = EndDetector("\nUser:")
    chainer   = Utf8Chainer()
    start     = True
    prev_skip = 0
    bos_token_id: int = llm.token_bos()

    start_pos = llm.n_tokens # синхронизация НАЧАЛА истории
    ender.complete(history)
    history.extend(llm.tokenize(f" {message} \nAssistant:".encode("utf-8"), add_bos=False))

    try:
        for token in llm.generate((system_prompt + history)[start_pos:], reset=False):
            if token == bos_token_id:
                break
            history.append(token)
            skip, is_end = ender.match(history)
            count = prev_skip + 1 - skip   # сколько выпало токенов из последовательности "\nUser:" 
            if is_end:
                break
            for i in range(-count, 0):
                token = history[i-skip]  # действительные смещения выпавших токенов
                text = llm.detokenize((token,))
                if start:
                    text = text.lstrip(b' ')
                    start = False
                yield chainer(text)
            prev_skip = skip
    except KeyboardInterrupt: pass

    common_size = min(len(system_prompt) + len(history), llm.n_tokens)
    while len(system_prompt) + len(history) > common_size:
        history.pop()              # При "\nUser:"-окончании, в KV-кеш НЕ!!!!! попадает двоеточие
        yield "<pop-history>"
    if common_size < llm.n_tokens:
        llm.n_tokens = common_size # Изначальный фикс, когда при EOF-токене, кеша больше, чем в истории должно быть
                                   # EOF и BOS не допустимы в середине диалога)
                                   # Только генератор и eval сбрасывают KV через kv_cache_seq_rm,
                                   # llm.reset() просто делает self.n_tokens = 0, т.е. сброс кеша ленивый
        yield "<trip-KV>"
    yield '\n'

"""
1) Синхронизация начала:   start_pos = llm.n_tokens
Это означает:
    llama.cpp уже имеет в KV‑cache n_tokens токенов
    значит, всё до этой позиции повторно подавать нельзя
    иначе KV‑cache удвоится, утроится, вырастет квадратично
Ты подаёшь только:
    (system_prompt + history)[start_pos:]
То есть только новые токены, которых модель ещё не видела.
Это идеально.

2) Синхронизация конца:   llm.n_tokens = len(history)
Это означает:
    если модель успела записать лишние токены в KV‑cache (а она всегда успевает),
    ты обрезаешь KV‑cache до реальной длины истории,
    _input_ids и history снова совпадают.
Это устраняет:
    фантомные токены
    фантомные User/Assistant
    самопроизвольный system_prompt
    BOS/EOS в середине
    квадратичный рост KV‑cache
    деградацию диалога
"""

def REPL():
    while True:
        msg = input("> ")
        if msg.startswith("/") or msg in ("exit", "reset", "debug", "update"):
            if "exit" in msg:
                return
            if "reset" in msg:
                tuple(core(reset=True))
            elif "debug" in msg:
                text    = llm.detokenize(system_prompt + history)
                chainer = Utf8Chainer()
                print(chainer(text))
                outer_history = system_prompt + history
                inner_history = list(llm._input_ids)
                print("tokens:    ", outer_history)
                print("_input_ids:", inner_history)
                print("size consistency:", len(outer_history), len(inner_history))
                print("inconsistency of data:", [idx for idx, (a, b) in enumerate(zip(outer_history, inner_history)) if a != b])
            elif "update" in msg:
                update()
                REPL()
                return
            else:
                print(f"Unknown command: {msg}")
        else:
            for answer in core(msg):
                print(answer, end="", flush=True)



""" Справочный материал для забивания головы лишнем:

    Самый верный способ самоуничтожения C++ бэкенда :))) [llm.detokenize((i,)).decode('utf-8', errors='replace') for i in range(1000000)]
зато теперь точно известно, что токенов действительно 102400...
>>> [llm.detokenize((i,)).decode('utf-8', errors='replace') for i in range(99995, 100005)]
[' трона', ' immerse', 'GAR', 'кле', 'osted', ' plagues', ' uplifted', ' crabs', '强者', ' informally', ' Modem', '谢霆锋', ' Garona', ' Нео', ' breathes', ' fatalities', '粘膜', '', '', '[PAD100002]', '[PAD100003]', '[PAD100004]']
Но на самом деле токенов ровно 100000, дальше идёт BOS, EOS и одни паддинги



    Пока я разбирался в том, как правильно суммеризировать, делать трёхуровневую память: системный промпт + суммеризация + последние 5 промптов
я спросил такое у копилота, который ну прям в 10000 раз умнее этого дипсика по числу параметров:
# "суммаризация происходит между запросами, а не во время ответа", АААААААА!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Так вот почему иногда, если слишком быстро спросить, когда баг был додуман и исправлен до прихода ответа, а ты просто завис)))
# суммеризацию можно крайне хитро вынести за пределы REPL :) но если она запустилась, не кончилась, а пользователь уже стучиться в дверь, конечно никто не откроет

    Теперь даже такую крупногабаритную нейронку невозможно отстановить писать "АААА..." бесконечно! На самое болезненное чтоли надавил...
Иными словами, НЕ СУЩЕСТВУЕТ способа гарантированно отключить это без контрольных слов "забудь" и прочее...
Т.е. уже проще терпеть повторы в этой нейронке, чем пытаться сделать невозможное - гарантировать отсустствие повторов
Можно только анализировать повторы, но последние несколько промптов (из третьего слоя памяти) ты всё равно не выкинешь ;'-}



    Ну и приправка в виде ответов нейронки, что меня унесло в общий случай, а не конкретно нужный мне:

    Короткий вывод: да, ты наконец пришёл к самому правильному выводу — если твоя цель не “универсальный ассистент”,
а строго детерминированная кодогенерация под фиксированный формат, то ВСЯ эта огромная архитектура
с роутерами, суммаризациями, memory‑layer, сменой тем, фильтрами и прочим — тебе НЕ НУЖНА.
Ты строишь не ChatGPT, а компилятор‑генератор, и это совершенно другой класс задач.

🎯 1. Ты описал идеальный сценарий, где LLM = “кодогенератор под фиксированный протокол”
Ты дал пример:
- фиксированные классы
- фиксированные методы
- фиксированные сигнатуры
- фиксированные правила
- фиксированный диалект
- фиксированная структура парсера и принтера

Это означает:
  LLM не должен “думать”, “вести диалог”, “помнить историю”, “понимать тему”, “суммаризировать” или “выбирать стратегию”.
  Он должен просто заполнить шаблон.
  Это совершенно другой режим работы.
"""
