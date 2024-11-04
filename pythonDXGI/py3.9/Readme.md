# CUDA 和深度学习库安装指南

## 概述

本指南旨在帮助用户在 Windows 上安装 NVIDIA CUDA、cuDNN，以及必要的深度学习库（TensorFlow、PyTorch 和 ONNX Runtime），以支持 Python 3.9.13。正确配置这些工具可以使你能够有效地开发和运行深度学习模型。

## 前提条件

- **操作系统**：Windows 10 或更高版本
- **Python**：3.9.13（确保在系统中已正确安装）
- **pip**：Python 包管理器（通常随 Python 一起安装）

## 安装步骤

### 1. 安装 NVIDIA CUDA

1. **下载 CUDA**：
   - 访问 [NVIDIA CUDA Toolkit 下载页面](https://developer.nvidia.com/cuda-downloads)。
   - 选择适合你的操作系统的 CUDA 版本（推荐 CUDA 11.6 或 CUDA 11.7）。

2. **运行安装程序**：
   - 双击下载的安装程序并按照提示进行安装。建议选择默认安装选项，以确保所有必要组件都被安装。

3. **设置环境变量**（如果未自动设置）：
   - 将以下路径添加到系统环境变量 `PATH` 中：
     ```
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\bin
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\libnvvp
     ```

### 2. 安装 cuDNN

1. **下载 cuDNN**：
   - 访问 [NVIDIA cuDNN 下载页面](https://developer.nvidia.com/cudnn)。
   - 需要创建一个 NVIDIA 开发者帐户并登录。
   - 下载与所安装的 CUDA 版本相对应的 cuDNN 版本（推荐 cuDNN 8.3.x）。

2. **安装 cuDNN**：
   - 解压下载的压缩文件。
   - 将解压后的 `bin`、`include` 和 `lib` 文件夹中的文件复制到 CUDA 安装目录下，通常为：
     ```
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\
     ```

### 3. 安装 Python 库

1. **创建虚拟环境**（可选，但推荐）：
   ```bash
   python -m venv myenv
   myenv\Scripts\activate  # Windows 上激活虚拟环境

### 4. 模型说明

- ID 分类：
  - 0: CT Body
  - 1: CT Head
  - 2: T Body
  - 3: T Head

