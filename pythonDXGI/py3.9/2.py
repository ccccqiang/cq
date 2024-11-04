from openvino.runtime import Core
import numpy as np
import cv2
import time

# 初始化 OpenVINO 的 Inference Engine
ie = Core()

# 加载 FP16 格式的模型
model_path = r"C:\Users\home123\Downloads\yolov8_openvino_model_fp16\test_fp16.xml"
compiled_model = ie.compile_model(model=model_path, device_name="CPU")

# 获取输入和输出的名称
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# 读取和预处理图像
image_path = r"C:\Users\home123\Desktop\OIP-C (1).jpg"
image = cv2.imread(image_path)
if image is None:
    raise ValueError("图像读取失败，请检查路径是否正确")

# 假设 YOLOv8 模型的输入大小为 640x640
input_size = (640, 640)

# 预处理图像
input_data = cv2.resize(image, input_size)  # 调整大小
input_data = input_data.transpose(2, 0, 1)  # 从 HWC 变为 CHW 格式
input_data = input_data[np.newaxis, ...]  # 添加批量维度
input_data = input_data.astype(np.float32) / 255.0  # 归一化到 [0, 1]

# 运行推理
start_time = time.time()
results = compiled_model([input_data])
end_time = time.time()

# 获取输出数据
output_data = results[output_layer]
print(f"推理时间: {end_time - start_time:.4f} 秒")
print(f"Output data shape: {output_data.shape}")
print(f"Output data sample: {output_data[0][:5]}")  # 显示输出数据的一部分样本

# 后处理函数 (YOLOv8 格式解码)
def process_yolo_output(output, image_shape, input_size):
    boxes, scores, class_ids = [], [], []
    
    # 处理每个检测结果
    for detection in output[0]:  # 遍历所有检测结果
        x_center, y_center, w, h, conf, class_id = detection[:6]

        if conf > 0.25:  # 过滤置信度低于 0.25 的结果
            # 转换为图像坐标
            h, w = image_shape[:2]
            x_center = x_center * w / input_size[0]
            y_center = y_center * h / input_size[1]
            w = w * w / input_size[0]
            h = h * h / input_size[1]

            x_min = int(x_center - w / 2)
            y_min = int(y_center - h / 2)
            x_max = int(x_center + w / 2)
            y_max = int(y_center + h / 2)

            # 确保坐标在图像边界内
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w - 1, x_max)
            y_max = min(h - 1, y_max)

            boxes.append([x_min, y_min, x_max, y_max])
            scores.append(conf)
            class_ids.append(int(class_id))

    return boxes, scores, class_ids

# 处理推理结果
boxes, scores, class_ids = process_yolo_output(output_data, image.shape, input_size)

# 打印检测结果到终端
print("Detected boxes:", boxes)
print("Detected scores:", scores)
print("Detected class IDs:", class_ids)

# 可视化检测结果
def draw_detections(image, boxes, scores, class_ids):
    for (box, score, class_id) in zip(boxes, scores, class_ids):
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, f"ID:{class_id} Score:{score:.2f}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 在原始图像上绘制检测结果
draw_detections(image, boxes, scores, class_ids)

# 显示结果
cv2.imshow('YOLOv8 Detection Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
