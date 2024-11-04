import cv2
import numpy as np
import torch
import onnxruntime as ort

# 查询库的版本
opencv_version = cv2.__version__
numpy_version = np.__version__
torch_version = torch.__version__
onnxruntime_version = ort.__version__

# 打印版本信息
print(f"opencv-python 版本: {opencv_version}")
print(f"numpy 版本: {numpy_version}")
print(f"torch 版本: {torch_version}")
print(f"onnxruntime 版本: {onnxruntime_version}")

# 生成 requirements.txt 文件
with open('requirements.txt', 'w') as f:
    f.write(f"opencv-python=={opencv_version}\n")
    f.write(f"numpy=={numpy_version}\n")
    f.write(f"torch=={torch_version}\n")
    f.write(f"onnxruntime=={onnxruntime_version}\n")

print("requirements.txt 文件已生成")
