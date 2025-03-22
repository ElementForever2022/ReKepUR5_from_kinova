@startuml

enum DebugColors {
    COLOR_BLACK
    COLOR_RED
    COLOR_GREEN
    COLOR_YELLOW
    COLOR_BLUE
    COLOR_PURPLE
    COLOR_CYAN
    COLOR_WHITE
}

' decorators to debug methods
class debug_decorator << (D,orchid) decorator >> {
    + head_message: str
    + tail_message: str
    + color_name: str
    + bold: bool
    + __call__(class_method: Callable)
    - wrapper(*args, **kwargs)
}

class print_debug << (F,lightgreen) function >> {
    + {static} print_debug(*args, color_name: str, bold: bool, end: str)
}

note right of debug_decorator
    decorator to debug methods
end note

note right of print_debug
    function to print debug messages
    support color and format
end note


DebugColors <-- debug_decorator: uses >
DebugColors <-- print_debug: uses >

note right of DebugColors
    colors supported by debug_decorator and print_debug
end note
@enduml