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

# 检查CUDA是否可用并设置执行提供者
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'CUDA' else ['CPUExecutionProvider']
onnx_model_path = r"C:\Users\home123\cq\pythonDXGI\py3.9\onnx\cs2.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=providers)

# 定义屏幕捕获区域（根据需要自定义区域）
g = DXGI.capture(0, 0, 320, 320)  # 从(0,0)到(320,320)区域进行捕获

# 定义计算FPS的变量
prev_time = 0
fps = 0

def preprocess(img):
    """
    将捕获的图像预处理为适合ONNX模型的输入。
    例如，调整大小、归一化、转为CHW格式等。
    """
    # 调整大小为 320x320，符合YOLOv5的输入要求
    img = cv2.resize(img, (320, 320))

    # 转为float32并归一化到0-1之间
    img = img.astype(np.float32) / 255.0

    # 交换维度为CHW格式
    img = np.transpose(img, (2, 0, 1))

    # 扩展维度以匹配batch格式
    img = np.expand_dims(img, axis=0)

    return img

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
    # 在这里处理输出并绘制边界框（需要根据YOLOv5的输出格式解析）

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
