import pyautogui
import pynput
import pyautogui
from simple_pid import PID
import ctypes
import time


def mouse_move(driver, target_x, target_y):
    """
    使用 PID 控制器平滑移动鼠标到目标坐标 (target_x, target_y)
    """
    mouse = pynput.mouse.Controller()
    pid_x = PID(0.065, 0.1, 0.01, setpoint=target_x)
    pid_y = PID(0.065, 0.1, 0.01, setpoint=target_y)

    while True:
        # 检查是否已接近目标坐标
        if abs(target_x - mouse.position[0]) < 3 and abs(target_y - mouse.position[1]) < 3:
            break

        # 使用 PID 控制器计算下一个位置
        next_x, next_y = pid_x(mouse.position[0]), pid_y(mouse.position[1])

        # 使用 driver 控制鼠标移动
        driver.moveR(int(round(next_x)), int(round(next_y)), False)

        # 使用 pyautogui 解决 pynput bug
        pyautogui.position()

        # 短暂延时，防止频繁计算导致过高的 CPU 占用
        time.sleep(0.01)


def move_mouse_to_head(coordinates, driver):
    # 假设coordinates的格式为 (x, y, class_id, confidence)
    # 仅移动到 CT Head (ID 1) 和 T Head (ID 3) 的坐标，并且置信度大于0.95
    head_coordinates = [(x, y) for (x, y, class_id, confidence) in coordinates if
                        class_id in [1] and confidence > 0.95]

    if not head_coordinates:
        print("没有符合条件的坐标！", head_coordinates)
        return

    for (x, y) in head_coordinates:
        print(f"正在移动鼠标到 ({x}, {y})")
        mouse_move(driver, x, y)

# from simple_pid import PID
# import pynput
# import pyautogui
# import ctypes
#
#
# def mouse_move(driver, target_x, target_y):
#     """
#     使用 PID 控制器平滑移动鼠标到目标坐标 (target_x, target_y)
#     """
#     mouse = pynput.mouse.Controller()
#     pid_x = PID(0.0035, 0.005, 0.0007, setpoint=target_x)
#     pid_y = PID(0.0035, 0.005, 0.0007, setpoint=target_y)
#
#     while True:
#         # 检查是否已接近目标坐标
#         if abs(target_x - mouse.position[0]) < 3 and abs(target_y - mouse.position[1]) < 3:
#             break
#
#         # 使用 PID 控制器计算下一个位置
#         next_x, next_y = pid_x(mouse.position[0]), pid_y(mouse.position[1])
#
#         # 使用 driver 控制鼠标移动
#         driver.moveR(int(round(next_x)), int(round(next_y)), False)
#
#         # 使用 pyautogui 解决 pynput bug
#         pyautogui.position()  # 此处用来触发 pyautogui 修复问题
#
#
# def move_mouse_to_head(coordinates, driver):
#     """
#     根据坐标列表移动鼠标到指定位置，只移动到 CT Head (ID 1) 和 T Head (ID 3) 的坐标，且置信度大于 0.95
#     """
#     # 筛选出符合条件的坐标
#     head_coordinates = [(x, y) for (x, y, class_id, confidence) in coordinates if
#                         class_id in [1, 3] and confidence > 0.95]
#
#     if not head_coordinates:
#         print("没有符合条件的坐标！",head_coordinates)
#         return
#
#     # 移动鼠标到每个筛选出来的坐标
#     for (x, y) in head_coordinates:
#         print(f"正在移动鼠标到 ({x}, {y})")
#         mouse_move(driver, x, y)