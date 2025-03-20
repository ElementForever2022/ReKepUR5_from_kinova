# from robotEnv import RobotEnv
from RobotEnvironment.robotEnv import RobotEnv
from RobotEnvironment.visualizer import Visualizer


import cv2
from typing import Callable
class RekepMain(Visualizer):
    def __init__(self, pc_id:int, instruction:str):
        super().__init__()
        self.env = RobotEnv(pc_id, warm_up=True)

        self.pc_id = pc_id
        self.instruction = instruction

        self.camera = self.env.global_camera

        self.add_keys(keys=['q'],
                      events=[self.__exit])
    
    def main_loop(self):
        self.keep_running = True
        while self.keep_running:
            self.set_screen_middle = self.camera.get_color_image()
            self.show()


    def add_keys(self, keys:list[str], events:list[Callable]):
        self.keys = keys
        self.events = events
    
    def __exit(self):
        try:
            self.keep_running = False
            # destroy the window if it exists
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow(self.window_name)
        except:
            pass # if the window does not exist, ignore the error


if __name__ == "__main__":
    instruction = 'help me grasp the cake'
    rekep_main = RekepMain(2, instruction)
    rekep_main.main_loop()