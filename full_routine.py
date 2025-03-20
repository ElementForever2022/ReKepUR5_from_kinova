# full routine of rekep
from photo import PRS
import cv2
from main_vision import MainVision

def main(instruction:str):
    # stage 1: take photo
    prs = PRS()
    while True:
        color_frame, depth_frame = prs.get_frames()
        cv2.imshow('frame', color_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            img = color_frame.copy()
            cv2.destroyAllWindows()
            break
    
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('instruction:', instruction)

    # save the imgs
    


    # stage 2: query LLM
    # python main_vision.py
    #  --instruction "help me grasp the rectangular cake and move up"
    #  --obj_list 'rectangular cake'
    #  --data_path /home/ur5/rekep/ReKepUR5_from_kinova/data --frame_number 2
    main_vision = MainVision(visualize=True)
    main_vision.perform_task()

if __name__ == "__main__":
    # main('help me grasp the cake')
    from RobotEnvironment import AutoCallibrator
    ac = AutoCallibrator(2,'global', (5,8), 0.0261)
    ac.callibration_loop()