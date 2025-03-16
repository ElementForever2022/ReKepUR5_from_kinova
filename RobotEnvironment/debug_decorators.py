"""
design the debug decorators for cameras and robotic arms
"""
# import necessary libs
import functools # to design decorator for methods
from typing import Callable # to specify input type as callable(function ptr)

# constants(colors of debug logs)
COLOR_BLACK         = "\033[30m"
COLOR_RED           = "\033[31m"
COLOR_GREEN         = "\033[32m"
COLOR_YELLOW        = "\033[33m"
COLOR_BLUE          = "\033[34m"
COLOR_PURPLE        = "\033[35m"
COLOR_CYAN          = "\033[36m"
COLOR_WHITE         = "\033[37m"
RESET               = "\033[0m" # reset color
BOLD                = "\033[1m" # bold font
# list of colors(type: str, values of colors)
COLORS = [
    COLOR_BLACK,
    COLOR_RED,
    COLOR_GREEN,
    COLOR_YELLOW,
    COLOR_BLUE,
    COLOR_PURPLE,
    COLOR_CYAN,
    COLOR_WHITE
]
# valid names of colors(type: str, names of colors)
__NAMES_OF_COLORS = [
    "COLOR_BLACK",
    "COLOR_RED",
    "COLOR_GREEN",
    "COLOR_YELLOW",
    "COLOR_BLUE",
    "COLOR_PURPLE",
    "COLOR_CYAN",
    "COLOR_WHITE"
]

# define decorator for methods
def debug_decorator(head_message:str='', tail_message:str='done', color_name:str='COLOR_WHITE', bold:bool=True) -> None:
    """
    demonstrate debug information

    inputs:
        - message:str, special message to print out(default null)
        - color_name:str(must in __NAMES_OF_COLORS), color of debug message
        - bold:bool(default True), whether outputlog is bold

    outputs:
        None
    """
    # check whether "color" is valid
    if color_name not in __NAMES_OF_COLORS:
        # raise invalid color excetion
        raise Exception(f'Invalid color:{color_name}. Valid color includes:{__NAMES_OF_COLORS}')
    # cvt color name into color ANSI api
    color = eval(color_name)
    
    # check whether output is bold
    if bold:
        bold_ansl = BOLD # bold ANSI instruction
    else:
        bold_ansl = '' # empty string

    def decorator(class_method: Callable):
        # wrap the whole class method in the decorator
        @functools.wraps(class_method)
        # define the debug messages of the decorator
        def wrapper(*args, **kwargs):
            if head_message=='':
                # default debug message
                
                # just before the method starts
                print(f"{color}{bold_ansl}[DEBUG] running {class_method.__name__}...{RESET}")
                # invoke the method
                result = class_method(*args, **kwargs)
                # just after the method ends
                print(f"{color}{bold_ansl}[DEBUG] {tail_message}{RESET}")
            else:
                # special debug message to show

                # just before the method starts
                print(f"{color}{bold_ansl}[DEBUG] {head_message}{RESET}")
                # invoke the method
                result = class_method(*args, **kwargs)
                # just after the method ends
                print(f"{color}{bold_ansl}[DEBUG] {tail_message}{RESET}")
            return result
        return wrapper
    return decorator

def print_debug(*args, color_name:str='COLOR_WHITE', bold:bool=True, end:str|None="\n"):
    """
    print a single line of debug message

    inputs:
        - color_name:str(must in __NAMES_OF_COLORS), color of debug message
        - bold:bool(default True), whether outputlog is bold
        - end:bool|None(default '\n'), whether to start a new line
    outputs:
        None
    """
    # check whether "color" is valid
    if color_name not in __NAMES_OF_COLORS:
        # raise invalid color excetion
        raise Exception(f'Invalid color:{color_name}. Valid color includes:{__NAMES_OF_COLORS}')
    # cvt color name into color ANSI api
    color = eval(color_name)

    # check whether output is bold
    if bold:
        bold_ansl = BOLD # bold ANSI instruction
    else:
        bold_ansl = '' # empty string

    # stack output string
    list_args_string = [f'{arg}' for arg in args]
    debug_string = ' '.join(list_args_string)

    if end==None:
        # print and start a new line
        print(f"{color}{bold_ansl}[DEBUG] {debug_string}{RESET}")
    else:
        # print and not start a new line
        print(f"{color}{bold_ansl}[DEBUG] {debug_string}{RESET}",end=f'{end}')



if __name__ == '__main__':
    import numpy as np
    print_debug('123','113',np.array([[12,12,12],[1,1,1]]), color_name='COLOR_RED')