# visualize the realtime camera images

# import the necessary packages
import cv2 # for image processing
import numpy as np # for numerical operations
from typing import Callable # for type hinting
from pynput import keyboard # for keyboard input
import abc # for abstract base classes

# import the necessary modules
from camera_manager import CameraManager # for camera management
from debug_decorators import debug_decorator, print_debug # for debugging

class Visualizer(object):
    """
    a class for visualizing the images
    """
    @debug_decorator(
        head_message='initializing visualizer...',
        tail_message='visualizer initialized!',
        color_name='COLOR_WHITE',
        bold=True
    )
    def __init__(self, height:int=480, width_left:int=640, width_right:int=640, width_middle:int=640, window_name:str='screen') -> None:
        """
        initialize the visualizer

        inputs:
            - height: the height of the screen
            - width_left: the width of the left screen
            - width_right: the width of the right screen
            - window_name: the name of the window
        """
        # resolution of the screen
        self.height = height
        self.width_left = width_left
        self.width_right = width_right
        self.width_middle = width_middle
        self.width = self.width_left + self.width_right + self.width_middle

        # initialize the screen
        self.screen = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # initialize the left and right screen
        self.screen_left = np.zeros((self.height, self.width_left, 3), dtype=np.uint8)
        self.screen_right = np.zeros((self.height, self.width_right, 3), dtype=np.uint8)
        self.screen_middle = np.zeros((self.height, self.width_middle, 3), dtype=np.uint8)
        self.screen_left_to_render = np.zeros((self.height, self.width_left, 3), dtype=np.uint8)
        self.screen_right_to_render = np.zeros((self.height, self.width_right, 3), dtype=np.uint8)
        self.screen_middle_to_render = np.zeros((self.height, self.width_middle, 3), dtype=np.uint8)
        
        # initialize the words queue
        self.words_queue = [] # list of words to be added to the screen
        self.words_background_queue = [] # list of backgrounds of words to be added to the screen


        # initialize the keys
        self.keys = []
        self.events = []

        # initialize the window
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        # end of initialization
    
    def __del__(self) -> None:
        """
        delete the visualizer
        """
        self.close()

    def add_keys(self, keys:list[str], events:list[Callable]) -> None:
        """
        add keys to the visualizer

        inputs:
            - keys: list[str], the keys to be initialized
            - events: list[Callable], the events to be triggered when the keys are pressed
        """
        self.keys.extend(keys)
        self.events.extend(events)

    def set_screen_left(self, screen:np.ndarray) -> None:
        """
        set the left screen

        inputs:
            - screen: np.ndarray (H,W,3), the left screen to be set
        """
        self.screen_left_to_render = screen
    


    def set_screen_right(self, screen:np.ndarray) -> None:
        """
        set the right screen

        inputs:
            - screen: np.ndarray (H,W,3), the right screen to be set
        """
        self.screen_right_to_render = screen
    
    def set_screen_middle(self, screen:np.ndarray) -> None:
        """
        set the middle screen

        inputs:
            - screen: np.ndarray (H,W,3), the middle screen to be set
        """
        self.screen_middle_to_render = screen

    def add_words(self, words:list[str]|str, screen_switch:str, position:tuple[int,int],
                  font_face:int=cv2.FONT_HERSHEY_SIMPLEX, font_scale:float=0.75,
                  color:tuple=(255,255,0), thickness:int=1,
                  padding:int=5, background_color:tuple=(0,0,0),
                  ) -> None:
        """
        add words to the screen

        inputs:
            - words: List[str]|str, the words to be added
            - screen_switch: str, the screen to be added to['left', 'right']
            - position: tuple (x,y), the position of the words
            - font_face: int, the font face(recommended: cv2.FONT_HERSHEY_SIMPLEX and leave default)
            - font_scale: float, the font scale
            - color: tuple (B,G,R), the color of the words
            - thickness: int, the thickness of the words
            - padding: int, the padding of the words
            - background_color: tuple (B,G,R), the background color of the words
        """
        if isinstance(words, str):
            words = [words]
        # calculate the text size
        (text_width, text_height), baseline = cv2.getTextSize(words[0], font_face, font_scale, thickness)
        
        # initialize the current positions of the words and the backgrounds
        current_words_position = (position[0], position[1] - text_height - 2*padding) # the top left of the first line of words
        # add the words to the queue
        for i, word in enumerate(words):
            # calculate the text size
            (text_width, text_height), _baseline = cv2.getTextSize(word, font_face, font_scale, thickness)
            # update the positions of the words and the backgrounds
            current_words_position = (current_words_position[0], current_words_position[1] + 2*(text_height*5//7))
            current_background_top_left = (current_words_position[0] - padding, current_words_position[1] -text_height*5//7-padding)
            current_background_bottom_right = (current_words_position[0] + text_width + padding, current_words_position[1] + text_height*5//7+padding)
            # add the positions to the lists
            self.words_queue.append((word, screen_switch, current_words_position,  font_face, font_scale, color, thickness))
            self.words_background_queue.append((screen_switch, current_background_top_left, current_background_bottom_right, background_color))

    def __render(self) -> None:
        """
        render the screen, add background to the screen and render the words
        """
        self.screen = np.zeros((self.height, self.width_left+self.width_right+self.width_middle, 3), dtype=np.uint8)
        self.screen_left = self.screen_left_to_render
        self.screen_right = self.screen_right_to_render
        self.screen_middle = self.screen_middle_to_render
        # render the backgrounds of the words
        for screen_switch, background_top_left, background_bottom_right, background_color in self.words_background_queue:
            if screen_switch == 'left':
                cv2.rectangle(self.screen_left, background_top_left, background_bottom_right, background_color, -1)
            elif screen_switch == 'right':
                cv2.rectangle(self.screen_right, background_top_left, background_bottom_right, background_color, -1)
            elif screen_switch == 'middle':
                cv2.rectangle(self.screen_middle, background_top_left, background_bottom_right, background_color, -1)
        # render the words
        for word, screen_switch, words_position, font_face, font_scale, color, thickness in self.words_queue:
            if screen_switch == 'left':
                cv2.putText(self.screen_left, word, words_position, font_face, font_scale, color, thickness)
            elif screen_switch == 'right':
                cv2.putText(self.screen_right, word, words_position, font_face, font_scale, color, thickness)
            elif screen_switch == 'middle':
                cv2.putText(self.screen_middle, word, words_position, font_face, font_scale, color, thickness)
        # add the screens to the screen
        self.screen[:,:self.width_left, :] = self.screen_left
        self.screen[:,self.width_left:self.width_left+self.width_middle, :] = self.screen_middle
        self.screen[:,self.width_left+self.width_middle:, :] = self.screen_right

        # empty the queues
        self.words_queue = []
        self.words_background_queue = []
        self.screen_left_to_render = np.zeros((self.height, self.width_left, 3), dtype=np.uint8)
        self.screen_right_to_render = np.zeros((self.height, self.width_right, 3), dtype=np.uint8)
        self.screen_middle_to_render = np.zeros((self.height, self.width_middle, 3), dtype=np.uint8)

    def show(self) -> None:
        """
        show the screen
        """
        # render the screen
        self.__render()
        # show the screen
        cv2.imshow(self.window_name, self.screen)
        # wait for 1ms
        key_pressed = cv2.waitKey(1) & 0xFF
        for key, event in zip(self.keys, self.events):
            if key_pressed == ord(key):
                event()
        # else:
        #     if key_pressed != 0xFF:
        #         print(f'key id:{key_pressed} not found, char={[chr(key_pressed)]}')

    def close(self) -> None:
        """
        close the window
        """
        try:
            # destroy the window if it exists
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow(self.window_name)
        except:
            pass # if the window does not exist, ignore the error
