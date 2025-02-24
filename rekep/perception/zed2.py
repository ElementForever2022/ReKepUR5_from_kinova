import pyzed.sl as sl
import cv2
import matplotlib.pyplot as plt

class ZED2Camera:
    def __init__(self, config):
        # Initialize ZED 2 camera
        self.config = config
        self.camera = sl.Camera()
        
        # Create camera parameters
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.AUTO  # 1280x720 by default
        self.init_params.camera_fps = 30  # Match config FPS
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.depth_minimum_distance = 0.3  # Min depth in meters
        self.init_params.depth_maximum_distance = 3.0  # Max depth in meters

        # Runtime parameters
        self.runtime_params = sl.RuntimeParameters()
        # self.runtime_params.sensing_mode = sl.SENSING_MODE_TYPE.STANDARD
        self.runtime_params.confidence_threshold = 100
        self.runtime_params.texture_confidence_threshold = 100

    def open(self):
        """Open the camera"""
        status = self.camera.open(self.init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"Camera open failed: {status}")
            return False
        return True

    def close(self):
        """Close the camera"""
        self.camera.close()

    def get_frame(self):
        """Get a new frame from camera"""
        image = sl.Mat()
        depth = sl.Mat()
        
        if self.camera.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.camera.retrieve_image(image, sl.VIEW.LEFT)
            self.camera.retrieve_measure(depth, sl.MEASURE.DEPTH)
            return image.get_data(), depth.get_data()
        return None, None

    def draw_frame(self, path, image, depth):
        """Save the frame in specific directory"""
        # Display images for debugging
        cv2.imshow("Image", image)
        cv2.imshow("Depth", depth)
        cv2.waitKey(1)  # Add small delay to allow display
        
        # Print debug info
        print(f"Image shape: {image.shape}")
        print(f"Depth shape: {depth.shape}")
        print(f"Saving to: {path}")
        plt.savefig(path)

    def enable_tracking(self):
        """Enable positional tracking"""
        tracking_params = sl.PositionalTrackingParameters()
        self.camera.enable_positional_tracking(tracking_params)

    def get_position(self):
        """Get camera position"""
        pose = sl.Pose()
        if self.camera.get_position(pose, sl.REFERENCE_FRAME.WORLD) == sl.POSE_STATUS.VALID:
            return pose.get_translation().get(), pose.get_rotation().get()
        return None, None

if __name__ == "__main__":
    # Create a basic config dictionary with default values
    config = {}
    cam = ZED2Camera(config)
    
    # Open camera first
    if cam.open():
        image, depth = cam.get_frame()
        cam.draw_frame("./data/zed2_test.jpg", image, depth)
    cam.close()
