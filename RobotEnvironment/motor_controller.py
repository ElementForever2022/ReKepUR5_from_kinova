# base class for motor control

# import the necessary libraries
import abc # for abstract class
import threading # for multi-threading
import time # for time operation


class RealtimeProperty(object):
    """
    a class to store the realtime property of the motor
    """
    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value

    def __str__(self):
        return f'{self.name}: {self.value}'

class MotorController(object):
    """
    a class to control the motor

    MUST implement the update method
    """
    def __init__(self, pc_id: int):
        """
        initialize the motor controller

        inputs:
            pc_id: int, the id of the pc, [1,2]
        """
        # set the ready state
        self.ready = False
        # check the pc id
        if pc_id not in [1,2]:
            raise ValueError(f'pc_id must be 1 or 2, got {pc_id}')
        # set the pc id
        self.pc_id = pc_id

        # initialize the realtime properties
        self.realtime_properties = {}

        # set a thread to update the realtime property
        self.keep_alive = True
        self.update_thread = threading.Thread(target=self.update_loop)
        self.update_thread.daemon = True # daemon thread
        self.update_thread.start() # start the thread

        
    
    def _add_realtime_property(self, name: str, value: float):
        """
        add a realtime property to the motor controller

        inputs:
            name: str, the name of the realtime property
            value: float, the value of the realtime property
        """
        self.realtime_properties[name] = RealtimeProperty(name, value)
    
    def _add_realtime_properties(self, name_list: list[str], value_list: list[float]):
        """
        add a list of realtime properties to the motor controller

        inputs:
            name_list: list[str], the names of the realtime properties
            value_list: list[float], the values of the realtime properties
        """
        for name, value in zip(name_list, value_list):
            self._add_realtime_property(name, value)
    
    def _get_realtime_property(self, name: str):
        """
        get a realtime property's value from the motor controller

        inputs:
            name: str, the name of the realtime property
        """
        return self.realtime_properties[name].value
    
    def _set_realtime_property(self, name: str, value: float):
        """
        set a realtime property's value to the motor controller

        inputs:
            name: str, the name of the realtime property
            value: float, the value of the realtime property
        """
        self.realtime_properties[name].value = value
    

    # overload the [] operator
    def __getitem__(self, name: str):
        return self._get_realtime_property(name)

    # overload the [] operator for setting the realtime property
    def __setitem__(self, name: str, value: float):
        self._set_realtime_property(name, value)
    
    def update_loop(self, period: float=0.01):
        """
        a loop to update the realtime property

        inputs:
            period: float, the period of the update loop, default is 0.1s
        """
        while self.keep_alive:
            if self.ready:
                self.update()
            time.sleep(period)

    # MUST implement the update method
    @abc.abstractmethod
    def update(self):
        """
        update the realtime property of the motor
        """
        raise NotImplementedError('update method is not implemented')
    
    def __del__(self):
        # stop the thread when the motor controller is deleted
        self.keep_alive = False
        self.update_thread.join(timeout=1.0)
        # pass
