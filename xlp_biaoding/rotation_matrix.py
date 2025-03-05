import numpy as np

def rotation_matrix_x(theta_x_deg):
    """绕 X 轴旋转"""
    theta_x = np.radians(theta_x_deg)  # 转换为弧度
    return np.array([[1, 0, 0],
                     [0, np.cos(theta_x), -np.sin(theta_x)],
                     [0, np.sin(theta_x), np.cos(theta_x)]])

def rotation_matrix_y(theta_y_deg):
    """绕 Y 轴旋转"""
    theta_y = np.radians(theta_y_deg)  # 转换为弧度
    return np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                     [0, 1, 0],
                     [-np.sin(theta_y), 0, np.cos(theta_y)]])

def rotation_matrix_z(theta_z_deg):
    """绕 Z 轴旋转"""
    theta_z = np.radians(theta_z_deg)  # 转换为弧度
    return np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                     [np.sin(theta_z), np.cos(theta_z), 0],
                     [0, 0, 1]])

def combined_rotation_matrix(theta_x_deg, theta_y_deg, theta_z_deg):
    """计算绕 X, Y, Z 轴的合成旋转矩阵"""
    Rx = rotation_matrix_x(theta_x_deg)
    Ry = rotation_matrix_y(theta_y_deg)
    Rz = rotation_matrix_z(theta_z_deg)
    
    # 按照 ZYX 顺序旋转：先绕 Z 轴，再绕 Y 轴，最后绕 X 轴
    R = Rz @ Ry @ Rx
    return R

# 测试：绕 X 轴 -90度，Y 轴 90度，Z 轴 45度的合成旋转
# theta_x = 73
# theta_y = 90
theta_z = -90

# R = combined_rotation_matrix(theta_x, theta_y, theta_z)
# print("合成旋转矩阵：")
# print(R)
Rx = rotation_matrix_z(theta_z)
R = np.array([
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0]
])
print(Rx)