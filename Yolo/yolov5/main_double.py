import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import win32api
import win32con
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.general import (LOGGER, check_img_size, cv2, non_max_suppression, xyxy2xywh, scale_coords)
from utils.torch_utils import select_device, time_sync
from grabscreen import ScreenGrabber
from PID import PID
import threading
from kalman import KalmanFilterWrapper
from mouse_controller import LogitechMouse,CH9350Mouse
# from FPS import FPS  # 导入FPS类
# 初始化FPS计数器
# fps = FPS()
# logitech_mouse = LogitechMouse()
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
def select_mouse(mouse_type="Logitech", port="COM6", baudrate=115200):
    """Return the appropriate mouse controller based on selection."""
    if mouse_type == "Logitech":
        return LogitechMouse()
    elif mouse_type == "CH9350":
        return CH9350Mouse(port=port, baudrate=baudrate)  # 传递参数
    else:
        raise ValueError("Unsupported mouse type. Choose 'Logitech' or 'CH9350'.")


def dynamic_speed(error, max_speed=50, min_speed=2, brake_zone=80, decay_factor=0.7):
    """
    :param error: 当前误差（目标位置-当前位置）
    :param max_speed: 最大移动步长
    :param min_speed: 最小移动步长
    :param brake_zone: 制动起始距离（像素）
    :param decay_factor: 阻尼衰减系数（0~1）
    :return: 带阻尼的速度值
    """
    abs_error = abs(error)

    # 死区处理
    if abs_error < 3:
        return 0

    # 基础速度曲线
    base_speed = max(min_speed, min(max_speed, abs_error * 0.5))

    # 制动区阻尼计算
    if abs_error < brake_zone:
        # 指数衰减：离目标越近减速越快
        damping = (abs_error / brake_zone) ** decay_factor
        base_speed *= damping

    return int(base_speed * (error / abs_error))


# 启动线程来更新 PID 参数
pid_update_thread = threading.Thread(target=update_pid_in_background, daemon=True)
pid_update_thread.start()
@torch.no_grad()
def find_target(
        # weights=ROOT / 'cs2_fp16.engine',  # model.pt path(s) 选择自己的模型
        weights=ROOT / r'C:\Users\Administrator\PycharmProjects\cq\onnx\Valorant_fp16.engine',  # model.pt path(s)
        # weights=ROOT / r'C:\Users\Administrator\PycharmProjects\cq\Yolo\yolov5\wazi_fp16.engine',  # model.pt path(s)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        # imgsz=(320, 320),  # inference size (height, width)
        imgsz=(416, 416),
        conf_thres=0.5,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=10,  # maximum detections per image
        device="0",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=[1],  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=True,  # class-agnostic NMS
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        use_capture_device = True,  # 设置为 True 表示使用采集卡
        device_index = 0,  # 采集卡索引，默认是0
        mouse_type="CH9350",  # Add mouse type selection here
):
    grabber = ScreenGrabber(use_capture_device=use_capture_device, device_index=device_index,)
    mouse_controller = select_mouse(mouse_type, port="COM6", baudrate=300000)
    kf_x = KalmanFilterWrapper(
        dt=137.2,
        process_noise=10,
        measurement_noise=10,
        initial_estimate=np.array([0, 0]),
        initial_covariance=np.eye(2)
    )

    kf_y = KalmanFilterWrapper(
        dt=137.2,
        process_noise=10,
        measurement_noise=10,
        initial_estimate=np.array([0, 0]),
        initial_covariance=np.eye(2)
    )


    global pause_aim, last_f1_state
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    bs = 1  # batch_size
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))
    time.sleep(0.01)

    # t1 = time_sync()

    # img0 = cv2.imread('./data/images/apex_test4.jpg')   # test picture
    # img0 = cv2.imread('./data/images/0.png')

    # for i in range(500):           # for i in range(500) 运行500轮测速 (run 500 rounds to check each round spend)
    print(f"imgz = {imgsz}")

    while True:
        current_f1_state = win32api.GetAsyncKeyState(win32con.VK_RSHIFT) & 0x8000
        if current_f1_state and not last_f1_state:
            pause_aim = not pause_aim
            print(f"Aim {'Stop' if pause_aim else 'Start'}")

        last_f1_state = current_f1_state

        if pause_aim:
            time.sleep(0.1)
            continue

        img0 = grabber.grab_screen(grab_window_location)
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
                    # final_x = target_xywh_x - screen_x_center
                    # final_y = target_xywh_y - screen_y_center - y_portion * target_xywh[3]
                    #
                    # pid_x = int(pid.calculate(final_x, 0))
                    # pid_y = int(pid.calculate(final_y, 0))
                    #
                    # # Move the mouse
                    # mouse_controller.move(pid_x, 0)
                    # # logitech_mouse.move(pid_x, pid_y)  # Call Logitech mouse move method
                    # print(f"Mouse-Move X Y = ({pid_x}, {pid_y})")
                    # 更新卡尔曼滤波器
                    kf_x.predict()
                    kf_y.predict()

                    # 使用卡尔曼滤波器更新目标的坐标
                    filtered_x = kf_x.update(target_xywh_x - screen_x_center)[0]
                    filtered_y = kf_y.update(target_xywh_y - screen_y_center - y_portion * target_xywh[3])[0]

                    final_x = int(filtered_x)
                    final_y = int(filtered_y)

                    # pid_x = int(pid.calculate(final_x, 0))
                    # pid_y = int(pid.calculate(final_y, 0))
                    raw_x = pid.calculate(final_x, 0)
                    raw_y = pid.calculate(final_y, 0)

                    # 应用动态速度曲线
                    pid_x = dynamic_speed(raw_x,
                                          max_speed=configs_dict[11],  # 使用配置的max_step
                                          min_speed=2,
                                          brake_zone=90,
                                          decay_factor=0.004)

                    pid_y = dynamic_speed(raw_y,
                                          max_speed=configs_dict[11],
                                          min_speed=2,
                                          brake_zone=60,
                                          decay_factor=0.0)

                    # 添加惯性衰减（模拟物理阻尼）
                    last_move = {'x': 0, 'y': 0}  # 在循环外初始化

                    # 惯性衰减计算
                    pid_x = int(pid_x * 0.7 + last_move['x'] * 0.3)
                    pid_y = int(pid_y * 0.7 + last_move['y'] * 0.3)
                    last_move.update({'x': pid_x, 'y': pid_y})

                    # 执行移动
                    if pid_x != 0 or pid_y != 0:
                        mouse_controller.move(pid_x, 0)
                        print(f"AdjSpeed: X{pid_x} Y{pid_y} | Raw: X{raw_x:.1f} Y{raw_y:.1f}")
                    # Move the mouse
                    # mouse_controller.move(pid_x, 0)
                    # logitech_mouse.move(pid_x, pid_y)  # Call Logitech mouse move method
                    # print(f"Mouse-Move X Y = ({pid_x}, {pid_y})")
        # else:
        #     print(f'No target found')
        # fps.update()
if __name__ == "__main__":
    find_target()
