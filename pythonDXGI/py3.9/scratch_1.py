import os
import sys
import time
import cv2
import numpy as np
import onnxruntime as ort
from ctypes import windll

# 确保所需的DLL在路径中
sys.path.append(r'C:\Users\Administrator\PycharmProjects\cq\pythonDXGI\py3.9')
os.add_dll_directory(r'C:\Users\Administrator\PycharmProjects\cq\pythonDXGI\py3.9\DXGI.pyd')

# Windows时间优化
windll.winmm.timeBeginPeriod(1)
stop = windll.kernel32.Sleep

# 导入 DXGI 屏幕捕获库
import DXGI

# 加载 ONNX 模型
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'CUDA' else ['CPUExecutionProvider']
onnx_model_path = r"C:\Users\Administrator\PycharmProjects\cq\pythonDXGI\py3.9\onnx\cs2.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=providers)

# 定义屏幕捕获区域
screen_width = 1920  # 设置为你的屏幕宽度
screen_height = 1080  # 设置为你的屏幕高度
capture_width = 320
capture_height = 320

# 捕获全屏
g = DXGI.capture(0, 0, screen_width, screen_height)

# 定义计算FPS的变量
prev_time = 0

# 预处理图像
def preprocess(img):
    img = cv2.resize(img, (320, 320))  # 调整图像为320x320
    img = img.astype(np.float32) / 255.0  # 归一化到0到1之间
    img = np.transpose(img, (2, 0, 1))  # 转换为 (C, H, W) 格式
    img = np.expand_dims(img, axis=0)  # 添加批量维度
    return img

# 后处理和NMS
def postprocess(output, img, conf_threshold=0.5, iou_threshold=0.4):
    boxes = []
    scores = []
    class_ids = []

    for detection in output[0]:
        x, y, w, h, conf, *class_scores = detection
        if conf > conf_threshold:
            class_id = np.argmax(class_scores)
            score = class_scores[class_id]
            if score > conf_threshold:
                boxes.append([x - w / 2, y - h / 2, w, h])  # 计算边界框
                scores.append(score)
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)

    detected_boxes = []

    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绘制矩形框
            label = f"{class_ids[i]}: {scores[i]:.2f}"
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 定义处理流程
while True:
    try:
        # 捕获全屏
        img = g.cap()

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # 转换为BGR格式

        # 计算中心位置的左上角坐标
        center_x = (screen_width - capture_width) // 2
        center_y = (screen_height - capture_height) // 2

        # 从全屏图像中提取中心320x320区域
        img_cropped = img[center_y:center_y + capture_height, center_x:center_x + capture_width]

        # 预处理图像
        input_tensor = preprocess(img_cropped)

        # ONNX 推理
        onnx_inputs = {ort_session.get_inputs()[0].name: input_tensor}
        onnx_outputs = ort_session.run(None, onnx_inputs)

        # 获取推理结果并绘制
        output = onnx_outputs[0]
        postprocess(output, img_cropped)

        # 显示带检测的图像
        cv2.imshow('YOLOv5n ONNX Detection', img_cropped)

        # 计算 FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        print(f"FPS: {fps:.2f}")

        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"发生错误: {e}")
        break

# 清理资源
cv2.destroyAllWindows()
