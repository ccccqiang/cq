import mss
from colorama import Fore, Style, init
import numpy as np
import math
import keyboard
import ctypes
import onnxruntime as ort
from simple_pid import PID
import pynput
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image
import time

max_age = 5  # 追踪对象的最大年龄
max_iou_distance = 0.7  # IOU阈值

deepsort = DeepSort(
    max_age=max_age,
    max_iou_distance=max_iou_distance,
)

CONFIDENCE_THRESHOLD = None
DETECTION_Y_PORCENT = None
init(autoreset=True)

# 加载ONNX模型
onnx_model_path = r"E:\123pan\Downloads\ai\onnx\cs2.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

def print_status(status_message):
    print(f"{Fore.CYAN}{status_message}{Style.RESET_ALL}")

def show_instructions():
    print(Fore.CYAN + "配置键:" + Style.RESET_ALL)
    print(Fore.YELLOW + "i: ct_head" + Style.RESET_ALL)
    print(Fore.YELLOW + "j: ct_body" + Style.RESET_ALL)
    print(Fore.YELLOW + "o: t_head" + Style.RESET_ALL)
    print(Fore.YELLOW + "k: t_body" + Style.RESET_ALL)
    print(Fore.YELLOW + "l: none" + Style.RESET_ALL)
    print(Fore.RED + "q: 退出" + Style.RESET_ALL)
    print(Fore.CYAN + "按下一个键..." + Style.RESET_ALL)

def convert_to_bbs(results, classes):
    bbs = []
    for obj in results[0]:  # 迭代第一维（即检测到的每个对象）
        # 假设最后三个值是 [x1, y1, x2, y2, conf, class_id]
        x1, y1, x2, y2, confidence, class_id = obj[-6], obj[-5], obj[-4], obj[-3], obj[-2], int(obj[-1])

        if CONFIDENCE_THRESHOLD is not None and confidence > CONFIDENCE_THRESHOLD and class_id in classes:
            bbox = [x1, y1, x2, y2]  # 这里假设你需要的是 [x1, y1, x2, y2]
            bbs.append((bbox, confidence, class_id))
            print(f"[DEBUG] Detected bbox: {bbox}, Confidence: {confidence}, Class ID: {class_id}")
    return bbs

def mouse_move(driver, target_x, target_y):
    mouse = pynput.mouse.Controller()
    while True:
        if abs(target_x - mouse.position[0]) < 3 and abs(target_y - mouse.position[1]) < 3:
            break
        pid_x = PID(0.25, 0.01, 0.01, setpoint=target_x)
        pid_y = PID(0.25, 0.01, 0.01, setpoint=target_y)
        next_x = pid_x(mouse.position[0])
        next_y = pid_y(mouse.position[1])
        driver.moveR(int(round(next_x)), int(round(next_y)), False)
        print(f"[DEBUG] Moving mouse to: ({next_x}, {next_y})")

def display_preview(frame, bbs):
    frame = np.transpose(frame, (1, 2, 0))  # 将形状转换为 (320, 320, 3)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 转换为 BGR 格式
    for bbox, confidence, class_id in bbs:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f'ID: {class_id}, Conf: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imshow('Preview', frame)
    print(f"[DEBUG] Number of bounding boxes: {len(bbs)}")

def main():
    driver = ctypes.CDLL(f'./logitech.driver.dll')

    with mss.mss() as sct:
        monitor_number = 1
        mon = sct.monitors[monitor_number]

        center_x = mon["left"] + mon["width"] // 2
        center_y = mon["top"] + mon["height"] // 2
        width = 320
        height = 320

        monitor = {
            "top": center_y - height // 2,
            "left": center_x - width // 2,
            "width": width,
            "height": height,
            "mon": monitor_number,
        }

        classes = [0]
        show_instructions()

        while True:
            img = np.array(Image.frombytes('RGB', (width, height), sct.grab(monitor).rgb))
            img = img.astype(np.float32) / 255.0  # 归一化图像
            img = np.transpose(img, (2, 0, 1))  # 将维度从 (H, W, C) 转换为 (C, H, W)
            img = np.expand_dims(img, axis=0)  # 添加批次维度

            # 使用ONNX模型进行推理
            ort_inputs = {ort_session.get_inputs()[0].name: img}
            results = ort_session.run(None, ort_inputs)

            bbs = convert_to_bbs(results[0], classes)
            trackers = deepsort.update_tracks(bbs, frame=img[0])
            largest_bbox = None
            largest_area = 0
            mouse_x, mouse_y = width / 2, height / 2
            nearest_distance = float('inf')
            max_conf = 0

            for track in trackers:
                bbox = track.to_tlwh()
                bbox = [int(coord) for coord in bbox]
                bbox_center_x = int(bbox[0] + bbox[2] / 2)
                bbox_center_y = int(bbox[1] + bbox[3] * DETECTION_Y_PORCENT)
                det_conf = track.det_conf
                distance = math.sqrt((bbox_center_x - mouse_x) ** 2 + (bbox_center_y - mouse_y) ** 2)

                if det_conf is not None and det_conf > max_conf:
                    area = bbox[2] * bbox[3]
                    if area > largest_area and distance < nearest_distance:
                        max_conf = det_conf
                        nearest_distance = distance
                        largest_area = area
                        largest_bbox = bbox

            if largest_bbox is not None:
                target_x = (largest_bbox[0] + largest_bbox[2] / 2)
                target_y = (largest_bbox[1] + largest_bbox[3] * DETECTION_Y_PORCENT)
                mouse_move(driver, target_x, target_y)

            display_preview(img[0], bbs)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.02)

            if keyboard.is_pressed('i'):
                classes = [2]
                CONFIDENCE_THRESHOLD = 0.8
                DETECTION_Y_PORCENT = 0.5
                print_status(f"[INFO] 配置: ct_head, confidence={CONFIDENCE_THRESHOLD}")
                continue
            if keyboard.is_pressed('j'):
                classes = [1]
                CONFIDENCE_THRESHOLD = 0.9
                DETECTION_Y_PORCENT = 0.1
                print_status(f"[INFO] 配置: ct_body, confidence={CONFIDENCE_THRESHOLD}")
                continue
            if keyboard.is_pressed('o'):
                classes = [4]
                CONFIDENCE_THRESHOLD = 0.8
                DETECTION_Y_PORCENT = 0.5
                print_status(f"[INFO] 配置: t_head, confidence={CONFIDENCE_THRESHOLD}")
                continue
            if keyboard.is_pressed('k'):
                classes = [3]
                CONFIDENCE_THRESHOLD = 0.9
                DETECTION_Y_PORCENT = 0.1
                print_status(f"[INFO] 配置: t_body, confidence={CONFIDENCE_THRESHOLD}")
                continue
            if keyboard.is_pressed('l'):
                classes = [0]
                CONFIDENCE_THRESHOLD = None
                DETECTION_Y_PORCENT = None
                print_status(f"[INFO] 配置: none")
                continue

if __name__ == "__main__":
    main()
