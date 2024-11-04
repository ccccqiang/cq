import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + package.split())

def main():
    # 确保 pip 是最新版本
    print("Upgrading pip...")
    install("pip --upgrade")

    # 安装 TensorFlow
    print("Installing TensorFlow...")
    install("tensorflow==2.9.0")

    # 安装 PyTorch
    print("Installing PyTorch...")
    install("torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu117")

    # 安装 ONNX Runtime
    print("Installing ONNX Runtime...")
    install("onnxruntime-gpu==1.10.2")

    print("All packages installed successfully!")

if __name__ == "__main__":
    main()
