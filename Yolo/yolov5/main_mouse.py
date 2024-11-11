import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import win32api
import win32con
import ctypes
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.general import (LOGGER, check_img_size, cv2, non_max_suppression, xyxy2xywh, scale_coords)
from utils.torch_utils import select_device, time_sync
from grabscreen import grab_screen
from PID import PID
# from FPS import FPS  # 导入FPS类
# 初始化FPS计数器
# fps = FPS()
# Load Logitech Driver DLL globally
try:
    driver = ctypes.CDLL(r"C:\Users\Administrator\PycharmProjects\cq\LGMC\logitech.driver.dll")
    ok = driver.device_open() == 1  # The driver can only be opened once per process
    if not ok:
        print('Error, GHUB or LGS driver not found')
except FileNotFoundError:
    print(f'Error, DLL file not found')

class Logitech:

    class mouse:
        """
        code: 1: Left button, 2: Middle button, 3: Right button
        """

        @staticmethod
        def press(code):
            if not ok:
                return
            driver.mouse_down(code)

        @staticmethod
        def release(code):
            if not ok:
                return
            driver.mouse_up(code)

        @staticmethod
        def click(code):
            if not ok:
                return
            driver.mouse_down(code)
            driver.mouse_up(code)

        @staticmethod
        def scroll(a):
            """
            a: Scroll step, unclear meaning
            """
            if not ok:
                return
            driver.scroll(a)

        @staticmethod
        def move(x, y):
            """
            Relative movement. For absolute movement, you need to use pywin32's win32gui to calculate positions.
            pip install pywin32 -i https://pypi.tuna.tsinghua.edu.cn/simple
            x: Horizontal movement distance and direction, positive to the right, negative to the left
            y: Vertical movement distance and direction
            """
            if not ok:
                return
            if x == 0 and y == 0:
                return
            driver.moveR(x, y, True)  # Relative movement

    class keyboard:
        """
        Keyboard key functions use the corresponding key code.
        code: 'a'-'z': A-Z, '0'-'9': 0-9, other keys are not specified
        """

        @staticmethod
        def press(code):
            if not ok:
                return
            driver.key_down(code)

        @staticmethod
        def release(code):
            if not ok:
                return
            driver.key_up(code)

        @staticmethod
        def click(code):
            if not ok:
                return
            driver.key_down(code)
            driver.key_up(code)

# Configuration and Constants
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# Load the configurations from file
com_text = ""
configs_dict = {}
config_list = []
with open('configs.txt', 'r', encoding="utf-8") as f:
    for config_line in f:
        config_list.append(list(config_line.strip('\n').split(',')))
f.close()

config_list.remove(['# 范围调节'])
config_list.remove(['# PID控制调节'])
for i in range(10):
    config_list.remove([''])
config_list.remove([''])

# Parse the configurations
index1 = config_list[0][0].find("=")
index2 = config_list[0][0].find("#")
com_text = config_list[0][0][index1 + 1:index2].strip()
del config_list[0]

last_configs_list = []
for i in range(len(config_list)):
    index1 = config_list[i][0].find("=")
    index2 = config_list[i][0].find("#")
    last_configs_list.append(float(config_list[i][0][index1 + 1:index2]))
    configs_dict[i + 1] = float(config_list[i][0][index1 + 1:index2])

y_correction_factor = configs_dict[1]
x_correction_factor = 0
screen_x, screen_y = configs_dict[2], configs_dict[3]
window_x, window_y = configs_dict[4], configs_dict[5]
screen_x_center = screen_x / 2
screen_y_center = screen_y / 2
PID_time = configs_dict[6]
Kp = configs_dict[7]
Ki = configs_dict[8]
Kd = configs_dict[9]
y_portion = configs_dict[10]
max_step = configs_dict[11]
pid = PID(PID_time, max_step, -max_step, Kp, Ki, Kd)

# Grab window location for screen capture
grab_window_location = (
    int(screen_x_center - window_x / 2 + x_correction_factor),
    int(screen_y_center - window_y / 2 - y_correction_factor),
    int(screen_x_center + window_x / 2 + x_correction_factor),
    int(screen_y_center + window_y / 2 - y_correction_factor))

edge_x = screen_x_center - window_x / 2
edge_y = screen_y_center - window_y / 2

# Aiming range settings
aim_x = configs_dict[13]
aim_x_left = int(screen_x_center - aim_x / 2)
aim_x_right = int(screen_x_center + aim_x / 2)

aim_y = configs_dict[14]
aim_y_up = int(screen_y_center - aim_y / 2 - y_correction_factor)
aim_y_down = int(screen_y_center + aim_y / 2 - y_correction_factor)
time.sleep(2)

@torch.no_grad()
def find_target():
    # Load model and set up inference
    weights = ROOT / 'cs2.engine'
    data = ROOT / 'data/coco128.yaml'
    imgsz = (320, 320)
    conf_thres = 0.5
    iou_thres = 0.45
    max_det = 10
    device = select_device(0)  # 选择设备，device 是 '0', 'cpu' 等字符串
    model = DetectMultiBackend(weights, device=torch.device(device), data=data, fp16=True)  # 确保传递 torch.device 对象
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    model.warmup(imgsz=(1, 3, *imgsz))
    time.sleep(0.5)

    while True:
        img0 = grab_screen(grab_window_location)
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGRA2BGR)

        img = letterbox(img0, imgsz, stride=stride, auto=pt)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if model.fp16 else img.float()
        img /= 255

        if len(img.shape) == 3:
            img = img[None]

        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)
        det = pred[0]

        target_distance_list = []
        target_xywh_list = []
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                target_xywh_list.append(xywh)
                target_distance = abs(edge_x + xywh[0] - screen_x_center)
                target_distance_list.append(target_distance)

            min_index = target_distance_list.index(min(target_distance_list))
            target_xywh = target_xywh_list[min_index]

            target_xywh_x = target_xywh[0] + edge_x
            target_xywh_y = target_xywh[1] + edge_y
            print('\033[0;33;40m' + f"target-X = {target_xywh_x}  target—Y = {target_xywh_y}" + '\033[0m')
            if aim_x_left < target_xywh_x < aim_x_right and aim_y_up < target_xywh_y < aim_y_down:

                aim_mouse = win32api.GetAsyncKeyState(win32con.VK_RBUTTON) \
                            or win32api.GetAsyncKeyState(win32con.VK_LBUTTON)

                if aim_mouse:
                    final_x = target_xywh_x - screen_x_center
                    final_y = target_xywh_y - screen_y_center - y_portion * target_xywh[3]

                    pid_x = int(pid.calculate(final_x, 0))
                    pid_y = int(pid.calculate(final_y, 0))

                    # Move the mouse
                    Logitech.mouse.move(pid_x, pid_y)  # Call Logitech mouse move method
                    print(f"Mouse-Move X Y = ({pid_x}, {pid_y})")

        else:
            print(f'No target found')
        # fps.update()
if __name__ == "__main__":
    find_target()
