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

def security_wrap(func):
    def wrapper(*a, **kw):
        try: return func(*a, **kw)
        except MySystemExit: raise
        except:
            for line in traceback.format_exc().splitlines():
                print(line, color="red")
            # raise MySystemExit
    return wrapper

def pprint(*a, **kv):
    print(pformat(*a, **kv))

def redraw_chain(range, cb, end=None):
    it = iter(range)
    @security_wrap
    def step():
        try: number = next(it)
        except StopIteration:
            if end is not None: end()
            return

        cb(number)

        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

        return bpy.app.timers.register(step)
    bpy.app.timers.register(step)

scene = bpy.context.scene

def frame_chain(range, cb):
    def step(frame):
        scene.frame_set(frame)
        cb(frame)
    def end():
        scene.frame_set(current)
    current = scene.frame_current
    redraw_chain(range, step, end)

# ~~~ codegup ~~~



@security_wrap
def solve():
    print("MEOW!")
    # exit("Упс", "ошибочка!", sep=", ")
    raise ValueError("Слишком громкое мяукание!")

solve()
