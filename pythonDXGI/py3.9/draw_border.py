import os
import sys
import time
import cv2
import numpy as np
import onnxruntime as ort
from ctypes import windll

# 确保所需的DLL在路径中
sys.path.append(r'C:\Users\home123\cq\pythonDXGI\py3.9')
os.add_dll_directory(r'C:\Users\home123\cq\pythonDXGI\py3.9\DXGI.pyd')

# Windows时间优化
windll.winmm.timeBeginPeriod(1)
stop = windll.kernel32.Sleep

# 导入 DXGI 屏幕捕获库
import DXGI

# 加载 ONNX 模型
onnx_model_path = r"E:\123pan\Downloads\ai\onnx\cs2.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# 定义屏幕捕获区域（根据需要自定义区域）
g = DXGI.capture(0, 0, 320, 320)  # 从(0,0)到(640,640)区域进行捕获

# 定义计算FPS的变量
prev_time = 0
fps = 0

def preprocess(img):
    """
    将捕获的图像预处理为适合ONNX模型的输入。
    例如，调整大小、归一化、转为CHW格式等。
    """
    # 调整大小为 640x640，符合YOLOv5的输入要求
    img = cv2.resize(img, (320, 320))

    # 转为float32并归一化到0-1之间
    img = img.astype(np.float32) / 255.0

    # 交换维度为CHW格式
    img = np.transpose(img, (2, 0, 1))

    # 扩展维度以匹配batch格式
    img = np.expand_dims(img, axis=0)

    return img

def postprocess(output, img, conf_threshold=0.5, iou_threshold=0.4):
    """
    解析模型输出并绘制边界框。
    """
    boxes = []
    scores = []
    class_ids = []

    for detection in output[0]:
        x, y, w, h, conf, *class_scores = detection
        if conf > conf_threshold:
            class_id = np.argmax(class_scores)
            score = class_scores[class_id]
            if score > conf_threshold:
                boxes.append([x - w / 2, y - h / 2, w, h])
                scores.append(score)
                class_ids.append(class_id)

    # NMS (Non-Maximum Suppression)
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)

    # 处理空列表的情况
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{class_id}: {score:.2f}"
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

while True:
    start_time = time.time()

    # 捕获屏幕
    capture_start = time.time()
    img = g.cap()
    img = np.array(img)
    capture_end = time.time()

    # 将图像转换为BGR格式（OpenCV兼容）
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # 预处理图像
    input_tensor = preprocess(img)

    # ONNX 推理
    inference_start = time.time()
    onnx_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    onnx_outputs = ort_session.run(None, onnx_inputs)
    inference_end = time.time()

    # 获取推理结果并绘制
    output = onnx_outputs[0]
    postprocess(output, img)

    # 显示带检测的图像
    cv2.imshow('YOLOv5n ONNX Detection', img)

    # 计算 FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # 打印时间信息
    capture_time = (capture_end - capture_start) * 1000  # 转换为毫秒
    inference_time = (inference_end - inference_start) * 1000  # 转换为毫秒
    total_time = (time.time() - start_time) * 1000  # 每帧的总时间

    print(f"FPS: {fps:.2f}, Capture Time: {capture_time:.2f} ms, Inference Time: {inference_time:.2f} ms, Total Time: {total_time:.2f} ms")

    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理资源
cv2.destroyAllWindows()
