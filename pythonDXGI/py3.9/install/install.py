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
    install("tensorflow==2.7")

    # 安装 PyTorch
    print("Installing PyTorch...")
    install("torch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html")

    # 安装 ONNX Runtime
    print("Installing ONNX Runtime...")
    install("onnxruntime-gpu==1.10.2")

    print("All packages installed successfully!")

if __name__ == "__main__":
    main()
