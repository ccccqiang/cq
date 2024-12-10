import torch
from pathlib import Path
from PIL import Image

# 模型路径
model_path = r"C:\Users\home123\cq\onnx\csbest.pt"

# 图像路径
image_path = r"C:\Users\home123\Downloads\_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=5890170185683978284&skey=@crypt_81641f91_487915f3b2e7bfe8431b5f83fa8f07c5&mmweb_appid=wx_webfilehelper.jpg"

# 加载 YOLOv5 模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)  # 加载自定义模型

# 加载图像
img = Image.open(image_path)

# 推理
results = model(img)

# 展示推理结果
results.show()  # 显示带有预测框的图像
results.save(Path("output"))  # 保存预测结果到指定目录
