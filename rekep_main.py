# from robotEnv import RobotEnv
from RobotEnvironment.robotEnv import RobotEnv

class RekepMain:
    def __init__(self, instruction:str):
        self.env = RobotEnv()

        self.instruction = instruction
    

if __name__ == "__main__":
    instruction = 'help me grasp the cake'
    rekep_main = RekepMain(instruction)