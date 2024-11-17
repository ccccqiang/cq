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
import threading
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
with open('Dconfig.txt', 'r', encoding="utf-8") as f:
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
print(f"配置写入：{configs_dict}")
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
classes = configs_dict[15]
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
# 暂停自瞄标志
pause_aim = False
last_f1_state = False
def load_config():
    """读取配置文件并更新 PID 控制参数"""
    global Kp, Ki, Kd, PID_time, pid, screen_x, screen_y, window_x, window_y, y_portion, classes

    with open('config_double.txt', 'r', encoding="utf-8") as f:
        config_list = []
        for config_line in f:
            # 移除行尾的空白字符并分割每行数据
            config_line = config_line.strip()
            if not config_line or config_line.startswith("#"):
                continue  # 跳过空行和注释行

            # 去除注释部分
            index_of_comment = config_line.find("#")
            if index_of_comment != -1:
                config_line = config_line[:index_of_comment].strip()  # 只保留代码部分

            # 如果这一行包含配置项（如 'key = value'），我们将其拆分并存储
            if '=' in config_line:
                config_list.append(config_line.split("="))

    # 打印读取的配置列表，调试用
    print(f"配置文件内容：{config_list}")

    # 确保配置文件中的 PID 参数行存在且格式正确
    try:
        # 解析 PID 参数
        Kp = float(config_list[7][1].strip())  # 假设 PID 参数在第 8 行，第 2 列
        Ki = float(config_list[8][1].strip())  # 第 9 行，第 2 列
        Kd = float(config_list[9][1].strip())  # 第 10 行，第 2 列
        PID_time = float(config_list[6][1].strip())  # PID 时间常数在第 7 行，第 2 列
        screen_x, screen_y  = float(config_list[2][1].strip()),float(config_list[3][1].strip())
        window_x, window_y = float(config_list[4][1].strip()),float(config_list[5][1].strip())
        y_portion = float(config_list[10][1].strip())
        classes = int(config_list[15][1].strip())
    except IndexError:
        # print("配置文件格式错误，无法解析 PID 参数。")
        return

    pid = PID(PID_time, max_step, -max_step, Kp, Ki, Kd)  # 更新 PID 控制器
    screen_x,screen_y = screen_x,screen_y
    y_portion = y_portion
    classes = classes
    # print(f"PID 参数更新为 Kp={Kp}, Ki={Ki}, Kd={Kd},{screen_x},{screen_y}")


def update_pid_in_background():
    """每隔一定时间更新一次 PID 参数"""
    while True:
        load_config()
        time.sleep(1)  # 每 1 秒钟更新一次

# 启动线程来更新 PID 参数
pid_update_thread = threading.Thread(target=update_pid_in_background, daemon=True)
pid_update_thread.start()
@torch.no_grad()
def find_target(
        weights=ROOT / 'cs2_fp16.engine',  # model.pt path(s) 选择自己的模型
        # weights=ROOT / r'C:\Users\home123\cq\pythonDXGI\py3.9\onnx\valorant-n-3.pt',  # model.pt path(s)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(320, 320),  # inference size (height, width)
        conf_thres=0.5,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=10,  # maximum detections per image
        device="0",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    load_config()
    with open('config_double.txt', 'r', encoding="utf-8") as f:
        config_list = []
        for config_line in f:
            # 移除行尾的空白字符并分割每行数据
            config_line = config_line.strip()
            if not config_line or config_line.startswith("#"):
                continue  # 跳过空行和注释行

            # 去除注释部分
            index_of_comment = config_line.find("#")
            if index_of_comment != -1:
                config_line = config_line[:index_of_comment].strip()  # 只保留代码部分

            # 如果这一行包含配置项（如 'key = value'），我们将其拆分并存储
            if '=' in config_line:
                config_list.append(config_line.split("="))
    if classes is None:
        classes = int(config_list[15][1].strip())
    global pause_aim, last_f1_state
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    bs = 1  # batch_size
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))
    time.sleep(0.5)

    # t1 = time_sync()

    # img0 = cv2.imread('./data/images/apex_test4.jpg')   # test picture
    # img0 = cv2.imread('./data/images/0.png')

    # for i in range(500):           # for i in range(500) 运行500轮测速 (run 500 rounds to check each round spend)
    print(f"imgz = {imgsz}")

    while True:
        current_f1_state = win32api.GetAsyncKeyState(win32con.VK_UP) & 0x8000
        if current_f1_state and not last_f1_state:
            pause_aim = not pause_aim
            print(f"Aim {'Stop' if pause_aim else 'Start'}")

        last_f1_state = current_f1_state

        if pause_aim:
            time.sleep(0.1)
            continue
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
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
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

                if configs_dict[12] == 3:
                    aim_mouse = win32api.GetAsyncKeyState(win32con.VK_RBUTTON) \
                                or win32api.GetAsyncKeyState(win32con.VK_LBUTTON)
                elif configs_dict[12] == 2:
                    aim_mouse = win32api.GetAsyncKeyState(win32con.VK_RBUTTON)

                elif configs_dict[12] == 1:
                    aim_mouse = win32api.GetAsyncKeyState(win32con.VK_LBUTTON)

                elif configs_dict[12] == 0:
                    aim_mouse = True

                else:
                    print("请填入正确的鼠标瞄准模式数字 1 或 2 或 3, Please fill the correct aim mod number 1 or 2 or 3")
                    break

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
