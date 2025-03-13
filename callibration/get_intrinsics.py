"""
get intrinsics of a Realsense camera from chessboard photos and its own api
"""

# import necessary libs
import numpy as np # matrix calculation
import cv2 # openCV

import pyrealsense2 as rs # camera

import time, datetime # get current time
import os, pathlib # file path
