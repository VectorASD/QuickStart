import bpy
from io import StringIO
import traceback
from pprint import pformat

_G = getattr(bpy, "_G", None)
if _G is None:
    _G = {}
    setattr(bpy, "_G", _G)

def get_console():
    scripting = bpy.data.screens["Scripting"]
    console = {a.type: a for a in scripting.areas}["CONSOLE"]
    window = {r.type: r for r in console.regions}["WINDOW"]

    context_override = bpy.context.copy()
    context_override.update({
        "scene": scripting,
        "area": console,
        "region": window,
    })
    return context_override

def printer(*arr, sep=" ", end="\n"):
    io = StringIO()
    write = io.write
    if arr:
        it = iter(arr)
        write(str(next(it)))
        for text in it:
            write(sep)
            write(str(text))
    # write(end)
    return io.getvalue()

console_context = get_console()
color2type = {"red": "ERROR", "green": "INFO", "blue": "OUTPUT", "white": "INPUT"}
_first = True

def print(*arr, sep=" ", end="\n", color="white"):
    type = color2type[color.lower()]
    with bpy.context.temp_override(**console_context):
        console = bpy.ops.console
        if _first:
            console.clear()
        for line in printer(*arr, sep=sep, end=end).split("\n"):
            console.scrollback_append(text=line, type=type)

print("~"*77, color="green")
_first = False

class MySystemExit(Exception): pass

def exit(*arr, sep=" ", end="\n", color="red"):
    print(*arr, sep=sep, end=end, color=color)
    raise MySystemExit

def pprint(*a, **kv):
    print(pformat(*a, **kv))

# ~~~ codegup ~~~



def solve():
    print("MEOW!")
    # exit("Упс", "ошибочка!", sep=", ")
    raise ValueError("Слишком громкое мяукание!")



# ~~~ codegup ~~~

try: solve()
except MySystemExit: pass
except:
    for line in traceback.format_exc().splitlines():
        print(line, color="red")
