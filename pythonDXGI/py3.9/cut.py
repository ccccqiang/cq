import os
import time
import sys
import cv2
import numpy as np
from ctypes import windll

# 添加所需的 DLL
# sys.path.append(r'C:\Users\home123\cq\pythonDXGI\py3.9')
# os.add_dll_directory(r'C:\Users\home123\cq\pythonDXGI\py3.9\DXGI.pyd')
sys.path.append(r'C:\Users\Administrator\PycharmProjects\cq\pythonDXGI\py3.9')
os.add_dll_directory(r'C:\Users\Administrator\PycharmProjects\cq\pythonDXGI\py3.9\DXGI.pyd')

# 初始化Windows定时器
windll.winmm.timeBeginPeriod(1)
stop = windll.kernel32.Sleep

# 导入DXGI和YOLO
import DXGI
# import torch

# 设置屏幕捕捉区域 (左上角到右下角)
g = DXGI.capture(0, 0, 320, 320)

# 计算帧数的辅助变量
prev_time = time.time()
fps_display_interval = 1  # 每秒显示一次帧率
fps_counter = 0  # 计数帧数
fps = 0  # 最终显示的帧率

while True:
    start_time = time.time()

    # 捕获屏幕
    img = g.cap()
    img = np.array(img)

    # 将图片从 BGRA 转换为 BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # 计算当前帧的时间
    frame_time = time.time() - start_time

    # 计算 FPS，每隔 1 秒显示一次
    fps_counter += 1
    if time.time() - prev_time >= fps_display_interval:
        fps = fps_counter / fps_display_interval
        fps_counter = 0
        prev_time = time.time()

    # 在终端输出帧数和每帧处理时间
    print(f"Frame Time: {frame_time * 1000:.2f} ms, FPS: {fps:.2f}")

    # 显示捕获的图像
    cv2.imshow('Screen Capture', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 关闭所有窗口
cv2.destroyAllWindows()
