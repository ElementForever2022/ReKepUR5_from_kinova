# fusion and registration for depth and rgb
# point cloud calibration
class MultiViewFusion:
    def __init__(self, config):
        self.zed = ZED2Camera(config)
        self.depth_pro = DepthProCamera(config)
        
    def fuse_point_clouds(self, frames):
        """Fuse multiple view point clouds"""
        point_clouds = []
        for frame in frames:
            pc = self.process_frame(frame)
            point_clouds.append(pc)
        return self.register_point_clouds(point_clouds)



import pyzed.sl as sl
import numpy as np
import cv2
import time
from scipy.spatial.transform import Rotation

class DualZEDSystem:
    def __init__(self):
        # Initialize camera objects
        self.cam1 = sl.Camera()
        self.cam2 = sl.Camera()
        
        # Camera parameters
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        self.init_params.coordinate_units = sl.UNIT.METER
        
        # Runtime parameters
        self.runtime_params = sl.RuntimeParameters()
        self.runtime_params.confidence_threshold = 50
        
    def connect_cameras(self):
        """Connect to both ZED cameras"""
        # Connect to first camera
        status1 = self.cam1.open(self.init_params)
        if status1 != sl.ERROR_CODE.SUCCESS:
            print(f"Camera 1 error: {status1}")
            return False
            
        # Connect to second camera with different ID
        self.init_params.input.set_from_serial_number(self.cam2.get_serial_number() + 1)
        status2 = self.cam2.open(self.init_params)
        if status2 != sl.ERROR_CODE.SUCCESS:
            print(f"Camera 2 error: {status2}")
            return False
            
        return True

    def calibrate_cameras(self):
        """Perform stereo calibration between the two cameras"""
        # Prepare calibration parameters
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Create calibration pattern
        pattern_size = (9, 6)  # Number of corners in the checkerboard
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints1 = []  # 2D points in image plane for camera 1
        imgpoints2 = []  # 2D points in image plane for camera 2
        
        # Capture calibration images
        for _ in range(20):  # Capture 20 different poses
            # Get images from both cameras
            image1 = sl.Mat()
            image2 = sl.Mat()
            
            if self.cam1.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                self.cam1.retrieve_image(image1, sl.VIEW.LEFT)
            if self.cam2.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                self.cam2.retrieve_image(image2, sl.VIEW.LEFT)
                
            # Convert to OpenCV format
            img1 = image1.get_data()
            img2 = image2.get_data()
            
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # Find checkerboard corners
            ret1, corners1 = cv2.findChessboardCorners(gray1, pattern_size, None)
            ret2, corners2 = cv2.findChessboardCorners(gray2, pattern_size, None)
            
            if ret1 and ret2:
                objpoints.append(objp)
                
                corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
                corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
                
                imgpoints1.append(corners1)
                imgpoints2.append(corners2)
                
            time.sleep(0.5)  # Wait for next pose
            
        # Perform stereo calibration
        ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints1, imgpoints2,
            self.cam1.get_calibration_parameters().left_cam.get_camera_matrix(),
            self.cam1.get_calibration_parameters().left_cam.get_distortion(),
            self.cam2.get_calibration_parameters().left_cam.get_camera_matrix(),
            self.cam2.get_calibration_parameters().left_cam.get_distortion(),
            gray1.shape[::-1], None, None,
            flags=cv2.CALIB_FIX_INTRINSIC
        )
        
        return R, T  # Return rotation and translation between cameras
        
    def get_keypoints(self, enable_body_tracking=True):
        """Get keypoints from both cameras and fuse them"""
        # Enable body tracking
        body_params = sl.BodyTrackingParameters()
        body_params.enable_tracking = True
        body_params.enable_body_fitting = True
        
        self.cam1.enable_body_tracking(body_params)
        self.cam2.enable_body_tracking(body_params)
        
        # Containers for body tracking
        bodies1 = sl.Bodies()
        bodies2 = sl.Bodies()
        
        # Get frames and track bodies
        if self.cam1.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.cam1.retrieve_bodies(bodies1)
        
        if self.cam2.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.cam2.retrieve_bodies(bodies2)
            
        # Extract keypoints
        keypoints1 = []
        keypoints2 = []
        
        for body in bodies1.body_list:
            keypoints1.extend(body.keypoint)
            
        for body in bodies2.body_list:
            keypoints2.extend(body.keypoint)
            
        return keypoints1, keypoints2
        
    def fuse_keypoints(self, keypoints1, keypoints2, R, T):
        """Fuse keypoints from both cameras using calibration data"""
        fused_keypoints = []
        
        for kp1, kp2 in zip(keypoints1, keypoints2):
            # Convert to homogeneous coordinates
            p1 = np.array([kp1[0], kp1[1], kp1[2], 1])
            p2 = np.array([kp2[0], kp2[1], kp2[2], 1])
            
            # Transform points to common coordinate system
            p2_transformed = np.dot(R, p2[:3]) + T.flatten()
            
            # Average the points
            fused_point = (p1[:3] + p2_transformed) / 2
            fused_keypoints.append(fused_point)
            
        return fused_keypoints
        
    def close(self):
        """Close both cameras"""
        self.cam1.close()
        self.cam2.close()

# Example usage
def main():
    dual_zed = DualZEDSystem()
    
    # Connect cameras
    if not dual_zed.connect_cameras():
        print("Failed to connect cameras")
        return
        
    # Calibrate cameras
    print("Performing calibration...")
    R, T = dual_zed.calibrate_cameras()
    print("Calibration complete")
    
    # Main loop
    try:
        while True:
            # Get keypoints from both cameras
            kp1, kp2 = dual_zed.get_keypoints()
            
            # Fuse keypoints
            fused_keypoints = dual_zed.fuse_keypoints(kp1, kp2, R, T)
            
            # Process or display fused keypoints as needed
            print(f"Number of fused keypoints: {len(fused_keypoints)}")
            
            time.sleep(0.1)  # Small delay to prevent maxing out CPU
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        dual_zed.close()

if __name__ == "__main__":
    main()