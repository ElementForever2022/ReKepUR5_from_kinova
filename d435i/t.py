import pyrealsense2 as rs

# 获取设备列表
context = rs.context()
devices = context.query_devices()

# 打印所有已连接的设备信息
print(f"Number of devices connected: {len(devices)}")
for i, dev in enumerate(devices):
    print(f"Device {i}: {dev.get_info(rs.camera_info.name)}")
    print(f"  Serial Number: {dev.get_info(rs.camera_info.serial_number)}")
