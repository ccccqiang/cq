import time
from simple_pid import PID
import ctypes
import pyautogui

# 尝试加载 DLL
try:
    gm = ctypes.CDLL(r'C:\Users\Administrator\PycharmProjects\cq\LGMC\ghub_device.dll')
    gmok = gm.device_open() == 1  # 判断驱动是否初始化成功
    if gmok:
        print("驱动初始化成功！")
    else:
        print("驱动初始化失败！")
except Exception as e:
    print(f"加载 DLL 时发生错误: {e}")
    gmok = False  # 如果 DLL 加载失败，则 gmok 为 False


# 按下鼠标按键
def press_mouse_button(button):
    if gmok:
        gm.mouse_down(button)


# 松开鼠标按键
def release_mouse_button(button):
    if gmok:
        gm.mouse_up(button)


# 点击鼠标按键
def click_mouse_button(button):
    press_mouse_button(button)
    release_mouse_button(button)


# 按下键盘按键
def press_key(code):
    if gmok:
        gm.key_down(code)


# 松开键盘按键
def release_key(code):
    if gmok:
        gm.key_up(code)


# 点击键盘按键
def click_key(code):
    press_key(code)
    release_key(code)


# 鼠标移动
def mouse_xy(x, y, abs_move=False):
    if gmok:
        gm.moveR(int(x), int(y), abs_move)


# 使用 PID 控制器平滑移动鼠标到目标坐标
def mouse_move(target_x, target_y):
    """
    使用 PID 控制器平滑移动鼠标到目标坐标 (target_x, target_y)
    """
    if not gmok:
        print("驱动未初始化，无法控制鼠标。")
        return

    # 修改这一行，获取 pyautogui 的鼠标控制器
    mouse = pyautogui

    # 设置PID控制器参数
    pid_x = PID(0.0037, 0.01, 0.0001, setpoint=target_x)
    pid_y = PID(0.0037, 0.01, 0.0001, setpoint=target_y)

    while True:
        # 检查是否已接近目标坐标
        if abs(target_x - mouse.position()[0]) < 3 and abs(target_y - mouse.position()[1]) < 3:
            break

        # 使用 PID 控制器计算下一个位置
        next_x, next_y = pid_x(mouse.position()[0]), pid_y(mouse.position()[1])

        # 使用 DLL 控制鼠标移动
        gm.moveR(int(round(next_x)), int(round(next_y)), False)

        # 短暂延时，防止频繁计算导致过高的 CPU 占用
        time.sleep(0.01)


# 仅移动到 CT Head (ID 1) 和 T Head (ID 3) 的坐标，并且置信度大于0.95
def move_mouse_to_head(coordinates):
    if not gmok:
        print("驱动未初始化，无法进行鼠标控制。")
        return

    # 假设coordinates的格式为 (x, y, class_id, confidence)
    head_coordinates = [(x, y) for (x, y, class_id, confidence) in coordinates if
                        class_id in [1, 3] and confidence > 0.95]

    if not head_coordinates:
        print("没有符合条件的坐标！", coordinates)  # 输出原始数据，便于调试
        return

    for (x, y) in head_coordinates:
        print(f"正在移动鼠标到 ({x}, {y})")
        mouse_move(x, y)

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