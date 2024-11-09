import cv2
import numpy as np
import onnxruntime
import subprocess

# 加载YOLO ONNX模型
onnx_model_path = r'C:\Users\home123\cq\pythonDXGI\py3.9\onnx\cs2.onnx'
ort_session = onnxruntime.InferenceSession(onnx_model_path)

# 定义YOLO推理函数
def yolo_inference(frame):
    # 将BGR图像转换为RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 进行YOLO推理
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    # 将输入图像调整为YOLO所需的尺寸
    resized_img = cv2.resize(img, (640, 640))  # 依赖于模型输入尺寸
    input_array = resized_img.transpose(2, 0, 1).astype(np.float32) / 255.0  # 归一化并调整维度
    input_array = np.expand_dims(input_array, axis=0)  # 扩展batch维度

    # 推理
    detections = ort_session.run([output_name], {input_name: input_array})[0]

    return detections

# 通过ffplay捕获屏幕流
ffplay_path = r"C:\Users\home123\Documents\screen-capture-record2dxgi演示\ffplay64.exe"
command = [
    ffplay_path,
    '-f', 'dshow',
    '-i', 'video=screen-capture-dxgi-qq35744025',  # 使用合适的设备名称
    '-x', '800',  # 屏幕分辨率
    '-vf', 'transpose=0,transpose=2',
    '-an', '-sn', '-t', '60'  # 可以设置捕获时长等
]

# 启动ffplay命令
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# 捕获视频流（使用OpenCV）
cap = cv2.VideoCapture('video=screen-capture-dxgi-qq35744025')  # 通过dshow捕获屏幕

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 对帧进行YOLO推理
    detections = yolo_inference(frame)

    # 假设detections包含bounding boxes: [x1, y1, x2, y2, confidence, class_id]
    for detection in detections:
        # 过滤低置信度的检测
        if detection[4] > 0.5:  # 置信度阈值
            x1, y1, x2, y2 = detection[:4]
            # 绘制边框
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # 显示推理后的帧
    cv2.imshow('YOLO Inference', frame)

    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
