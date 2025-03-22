@startuml

abstract class Visualizer{
+ height
+ width
+ width_left/middle/right
+ screen
+ screen_left/middle/right
+ screen_left/middle/right_to_render

+ words_queue = [
    tuple[wordstr,scr_enum, pos, font, ...]
]
+ words_background_queue

+ keys = [char]
+ events = [funcptr]
----
+ __init__() # set up the obj
+ __del__() # delete the
+ add_keys(keys=[str], events=[funcptr]) 
+ set_screen_left/middle/right(screen=arr)
+ add_words(words=[str],scr_enum,pos,...)
-  __render()
+ show()
+ close()
}



class RealtimeProperty {
    + name: str
    + value: float
    + __init__(name: str, value: float)
    + __str__(): str
}

abstract class MotorController {
    + ready: bool
    + pc_id: int
    + realtime_properties: dict
    + keep_alive: bool
    + update_thread: Thread
    
    + __init__(pc_id: int)
    + _add_realtime_property(name: str, value: float)
    + _add_realtime_properties(name_list: list[str], value_list: list[float])
    + _get_realtime_property(name: str): float
    + _set_realtime_property(name: str, value: float)
    + __getitem__(name: str): float
    + __setitem__(name: str, value: float)
    + update_loop(period: float)
    + {abstract} update() # must be implemented
    + __del__()
}

' camera related class
class RealsenseCamera {
  - serial_number: str
  - pipeline: rs.pipeline
  - width: int
  - height: int
  - fps: int
  - intrinsics: rs.intrinsics
  + __init__(serial_number: int, width: int, height: int, fps: int)
  + get_color_image(): np.ndarray
  + get_depth_image(): np.ndarray
  + launch_realtime_viewer(exit_key: str)
  - __get_frame(): tuple
}

class CameraManager {
  - pc_id: int
  - width: int
  - height: int
  - fps: int
  - camera_dict: dict
  + __init__(pc_id: int, width: int, height: int, fps: int)
  + get_camera(camera_position: str): RealsenseCamera
  + get_global_color_image(): np.ndarray
  + get_global_depth_image(): np.ndarray
  + get_wrist_depth_image(): np.ndarray
}

' visualization related class
'class Visualizer {
'  - height: int
'  - width_left: int
'  - width_right: int
'  - width_middle: int
'  - window_name: str
'  - screen: np.ndarray
'  + __init__(height: int, width_left: int, width_right: int, 'width_middle: int, window_name: str)
'  + add_words(words: list[str], screen_switch: str, position: tuple)
'  + show()
'  + close()
'}

' robot environment class
class RobotEnvironment {
  - gripper: Gripper
  - camera_manager: CameraManager
  - controller: rtde.RTDE
  + __init__(pc_id: int)
  + __move_to_pose(pose: list[float], trajectory_time: float)
  + update()
  - __get_tcp_pose(): list[float]
  - __get_joint_positions(): list[float]
}

' gripper class
class Gripper {
  - ser: serial.Serial
  - gripper_state: float
  + __init__(pc_id: int)
  + open()
  + close()
  + grasp()
}

class IntrinsicsCalculator {
    -pc_id: int
    -camera: RealsenseCamera
    -chessboard_shape: tuple
    -square_size: float
    -save_path: str
    +__init__(pc_id: int, chessboard_shape: tuple, square_size: float)
    +shoot_loop()
    +_calculate_intrinsics(save: bool)
    -__shoot()
    -__exit()
    -__empty_cache()
}

class AutoCallibrator {
    -camera_intrinsics: ndarray
    -calculated_intrinsics_matrix: ndarray
    -position_x: float
    -position_y: float
    -robot2camera: ndarray
    +__init__(pc_id: int)
    +callibration_loop()
    -__callibrate()
    -__save_matrix()
    +pixel_to_robot(u: int, v: int, Z: int)$
}


'MotorController o-- RealtimeProperty : contains >

MotorController <|-- RobotEnvironment
Visualizer <|-- RobotEnvironment
MotorController <|-- Gripper
Visualizer <|-- IntrinsicsCalculator
IntrinsicsCalculator <|-- AutoCallibrator
RobotEnvironment *-- Gripper
RobotEnvironment *-- CameraManager
CameraManager *-- RealsenseCamera
'RobotEnvironment *-- MotorController
MotorController *-- RealtimeProperty
'Gripper *-- MotorController

note right of MotorController
  abstract class to control motors
  must implement update method
end note

note bottom of RealtimeProperty
  struct to store motor's realtime properties
  that can be be updated FREQUENTLY
end note

note right of Visualizer
  abstract class to visualize
end note


@enduml
