import ctypes
import time

# 加载 Logitech 驱动 DLL
driver = ctypes.CDLL(r'C:\Users\home123\cq\LGMC\logitech.driver.dll')  # 替换为 Logitech 驱动 DLL 的实际路径

# 定义鼠标移动函数，假设 Logitech DLL 提供 `MoveMouse` 函数
def move_mouse(x, y):
    try:
        driver.MoveMouse(ctypes.c_int(x), ctypes.c_int(y))  # 调用 Logitech 驱动中的鼠标移动函数
    except AttributeError:
        print("The Logitech driver does not contain the MoveMouse function.")

def move_mouse_to_head(coordinates):
    """
    移动鼠标到检测到的头部位置。
    coordinates: list, 每个元素为 (x, y, class_id, confidence)
    """
    head_coordinates = [(x, y) for (x, y, class_id, confidence) in coordinates if class_id in [1, 3] and confidence > 0.95]

    for (x, y) in head_coordinates:
        # 移动鼠标到指定位置
        move_mouse(x, y)
        time.sleep(0.1)  # 短暂延迟以使移动流畅
