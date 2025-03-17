# visualize the realtime camera images

# import the necessary packages
import cv2 # for image processing
import numpy as np # for numerical operations
from typing import Callable # for type hinting
from pynput import keyboard # for keyboard input

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
    def __init__(self, height:int=480, width_left:int=640, width_right:int=640, window_name:str='screen') -> None:
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
        self.width = width_left + width_right

        # initialize the screen
        self.screen = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # initialize the left and right screen
        self.screen_left = np.zeros((self.height, self.width_left, 3), dtype=np.uint8)
        self.screen_right = np.zeros((self.height, self.width_right, 3), dtype=np.uint8)
        
        # initialize the words queue
        self.words_queue = [] # list of words to be added to the screen
        self.words_background_queue = [] # list of backgrounds of words to be added to the screen

        # initialize the window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.width, self.height)
        # end of initialization
    
    def __del__(self) -> None:
        """
        delete the visualizer
        """
        self.close()

    def init_key(self, key:str, event:Callable) -> None:
        """
        initialize the key

        inputs:
            - key: str, the key to be initialized
            - event: Callable, the event to be triggered when the key is pressed
        """
        def on_press(k):
            try:
                if k.char == key:
                    # execute the event
                    event()
                    return False
            except AttributeError:
                print(f'special key {k} is pressed')
        def on_release(k):
            if k.char == key:
                return False

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    def set_screen_left(self, screen:np.ndarray) -> None:
        """
        set the left screen

        inputs:
            - screen: np.ndarray (H,W,3), the left screen to be set
        """
        self.screen_left = screen

    def set_screen_right(self, screen:np.ndarray) -> None:
        """
        set the right screen

        inputs:
            - screen: np.ndarray (H,W,3), the right screen to be set
        """
        self.screen_right = screen

    def add_words(self, words:list[str]|str, screen_switch:str, position:tuple[int,int],
                  font_face:int=cv2.FONT_HERSHEY_SIMPLEX, font_scale:float=0.5,
                  color:tuple=(255,255,0), thickness:int=1,
                  padding:int=10, background_color:tuple=(0,0,0),
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
        # calculate the positions of the words as well as the backgrounds
        begining_top_left = (position[0], position[1] - text_height - 2*padding) # the top left of the first line of words
        begining_bottom_right = (position[0] + text_width + padding, position[1] + padding - text_height) # the bottom right of the first line of words
        # initialize the positions of the words and the backgrounds
        words_top_lefts = []
        words_bottom_rights = []
        background_top_lefts = []
        background_bottom_rights = []
        # initialize the current positions of the words and the backgrounds
        current_words_top_left = begining_top_left
        current_words_bottom_right = begining_bottom_right
        current_background_top_left = begining_top_left
        current_background_bottom_right = begining_bottom_right
        # add the words to the queue
        for i, word in enumerate(words):
            # calculate the text size
            (text_width, text_height), baseline = cv2.getTextSize(word, font_face, font_scale, thickness)
            # update the positions of the words and the backgrounds
            current_words_top_left = (current_words_top_left[0], current_words_top_left[1] + text_height + padding*2)
            current_words_bottom_right = (current_words_top_left[0] + text_width, current_words_top_left[1] + text_height)
            current_background_top_left = (current_words_top_left[0] - padding, current_words_top_left[1] - padding)
            current_background_bottom_right = (current_words_bottom_right[0] + padding, current_words_bottom_right[1] + padding)
            # add the positions to the lists
            words_top_lefts.append(current_words_top_left)
            words_bottom_rights.append(current_words_bottom_right)
            background_top_lefts.append(current_background_top_left)
            background_bottom_rights.append(current_background_bottom_right)
            self.words_queue.append((word, screen_switch, current_words_top_left, current_words_bottom_right, font_face, font_scale, color, thickness))
            self.words_background_queue.append((screen_switch, current_background_top_left, current_background_bottom_right, background_color))
    
    def __render(self) -> None:
        """
        render the screen, add background to the screen and render the words
        """

        # render the backgrounds of the words
        for screen_switch, background_top_left, background_bottom_right, background_color in self.words_background_queue:
            if screen_switch == 'left':
                cv2.rectangle(self.screen_left, background_top_left, background_bottom_right, background_color, -1)
            elif screen_switch == 'right':
                cv2.rectangle(self.screen_right, background_top_left, background_bottom_right, background_color, -1)
        # render the words
        for word, screen_switch, words_top_left, words_bottom_right, font_face, font_scale, color, thickness in self.words_queue:
            if screen_switch == 'left':
                cv2.putText(self.screen_left, word, words_top_left, font_face, font_scale, color, thickness)
            elif screen_switch == 'right':
                cv2.putText(self.screen_right, word, words_top_left, font_face, font_scale, color, thickness)
        # add the screens to the screen
        self.screen[:,:self.width_left, :] = self.screen_left
        self.screen[:,self.width_left:, :] = self.screen_right
    
    def show(self) -> None:
        """
        show the screen
        """
        # render the screen
        self.__render()
        # show the screen
        cv2.imshow(self.window_name, self.screen)
        # wait for 1ms
        cv2.waitKey(1)

    def close(self) -> None:
        """
        close the window
        """
        cv2.destroyWindow(self.window_name)
