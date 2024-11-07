import os
import sys
import time
import cv2
import numpy as np
import torch
from ctypes import windll
from mouse_controller import move_mouse_to_head
import ctypes
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# 添加DLL路径
sys.path.append(r'C:\Users\Administrator\PycharmProjects\cq\pythonDXGI\py3.9')
os.add_dll_directory(r'C:\Users\Administrator\PycharmProjects\cq\pythonDXGI\py3.9\DXGI.pyd')

# Windows 时间优化
windll.winmm.timeBeginPeriod(1)
stop = windll.kernel32.Sleep

# 导入 DXGI 屏幕捕获库
import DXGI

# 初始化 TensorRT 运行时
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(TRT_LOGGER)

# 加载 TensorRT 引擎文件
onnx_model_path = r"C:\Users\Administrator\PycharmProjects\cq\pythonDXGI\py3.9\onnx\cs2_fp16.engine"
with open(onnx_model_path, 'rb') as f:
    engine_data = f.read()
engine = runtime.deserialize_cuda_engine(engine_data)

# 创建执行上下文
context = engine.create_execution_context()

# 获取输入和输出形状
input_shape = None
output_shapes = {}

for i in range(engine.num_io_tensors):
    tensor_name = engine.get_tensor_name(i)
    tensor_mode = engine.get_tensor_mode(tensor_name)
    tensor_shape = engine.get_tensor_shape(tensor_name)
    if tensor_mode == trt.TensorIOMode.INPUT:
        input_shape = tensor_shape
    elif tensor_mode == trt.TensorIOMode.OUTPUT:
        output_shapes[tensor_name] = tensor_shape

if input_shape is None or not output_shapes:
    raise ValueError("无法找到输入或输出形状")

# 分配输入和输出缓冲区
input_data = np.random.rand(*input_shape).astype(np.float32)
output_buffers = {name: np.empty(shape, dtype=np.float32) for name, shape in output_shapes.items()}

d_input = cuda.mem_alloc(input_data.nbytes)
d_outputs = {name: cuda.mem_alloc(buffer.nbytes) for name, buffer in output_buffers.items()}

bindings = [int(d_input)] + [int(d_outputs[name]) for name in output_shapes]

# 定义屏幕捕获参数
screen_width = 1920
screen_height = 1080
capture_width = 320
capture_height = 320
g = DXGI.capture(0, 0, screen_width, screen_height)

# FPS 计算
prev_time = 0

def preprocess(img):
    img = cv2.resize(img, (320, 320)).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def postprocess(output, img, conf_threshold=0.5, iou_threshold=0.4):
    boxes, scores, class_ids, detected_boxes = [], [], [], []

    for detection in output[0]:
        # 打印检测的每个值，检查其结构
        # print(f"detection: {detection}")

        # 检查 detection 的长度是否符合预期
        if len(detection) >= 5:
            x, y, w, h, conf, *class_scores = detection
            if conf > conf_threshold:
                class_id = np.argmax(class_scores)
                score = class_scores[class_id]
                if score > conf_threshold:
                    boxes.append([x - w / 2, y - h / 2, w, h])
                    scores.append(score)
                    class_ids.append(class_id)
        # else:
            # print(f"Warning: detection has less than 5 values, skipping this detection: {detection}")

    # 应用非极大抑制（NMS）来去除冗余框
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)

        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x, y, w, h = map(int, box)
                class_id = class_ids[i]
                confidence = scores[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{class_id}: {confidence:.2f}"
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                detected_boxes.append((x + w // 2, y + h // 2, class_id, confidence))

    return detected_boxes

while True:
    try:
        img = g.cap()
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # 居中裁剪屏幕图像
        center_x, center_y = (screen_width - capture_width) // 2, (screen_height - capture_height) // 2
        img_cropped = img[center_y:center_y + capture_height, center_x:center_x + capture_width]

        # 预处理图像并执行 TensorRT 推理
        input_tensor = preprocess(img_cropped)

        # 确保输入数组是连续的
        input_tensor = np.ascontiguousarray(input_tensor)

        cuda.memcpy_htod(d_input, input_tensor)

        context.execute_v2(bindings)

        for name, buffer in output_buffers.items():
            cuda.memcpy_dtoh(buffer, d_outputs[name])

        output = list(output_buffers.values())

        # 后处理输出并绘制结果
        detected_boxes = postprocess(output, img_cropped)

        # 调整坐标以适应全屏
        adjusted_boxes = [(center_x + x, center_y + y, class_id, confidence) for (x, y, class_id, confidence) in detected_boxes]
        # print(adjusted_boxes)

        # 调用鼠标移动函数
        driver = ctypes.CDLL(r'C:\Users\Administrator\PycharmProjects\cq\LGMC\logitech.driver.dll')  # 加载驱动
        move_mouse_to_head(adjusted_boxes, driver)

        # 显示检测结果
        cv2.imshow('YOLOv5n TensorRT Detection', img_cropped)

        # FPS 计算
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        print(f"FPS: {fps:.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"An error occurred: {e}")
        break

# 清理
cv2.destroyAllWindows()
