# CUDA 和深度学习库安装指南

## 概述

本指南帮助你在 Windows 上安装 CUDA、cuDNN 以及必要的深度学习库（TensorFlow、PyTorch 和 ONNX Runtime），以支持 Python 3.9.13。

## 前提条件

- Windows 操作系统
- Python 3.9.13
- pip（Python 包管理器）

## 安装步骤

### 1. 安装 CUDA

1. **下载 CUDA**：
   - 访问 [NVIDIA CUDA Toolkit 下载页面](https://developer.nvidia.com/cuda-downloads)。
   - 选择适合你的操作系统的 CUDA 版本（推荐 CUDA 11.6 或 11.7）。

2. **运行安装程序**：
   - 双击下载的安装程序并按照说明进行安装。建议选择默认选项。

### 2. 安装 cuDNN

1. **下载 cuDNN**：
   - 访问 [NVIDIA cuDNN 下载页面](https://developer.nvidia.com/cudnn)。
   - 选择与安装的 CUDA 版本相对应的 cuDNN 版本（推荐 cuDNN 8.3.x）。

2. **安装 cuDNN**：
   - 解压下载的文件。
   - 将解压后的 `bin`、`include` 和 `lib` 文件夹中的文件复制到 CUDA 安装目录下：
     - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\`

### 3. 安装 Python 库

1. **创建虚拟环境**（可选）：
   ```bash
   python -m venv myenv
   myenv\Scripts\activate  # 激活虚拟环境
