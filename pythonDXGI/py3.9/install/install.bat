@echo off
echo Installing CUDA and cuDNN

:: 设置 CUDA 和 cuDNN 的下载链接（根据你的需求进行调整）
set CUDA_INSTALLER=https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_windows.exe
set CUDNN_INSTALLER=https://developer.nvidia.com/rdp/cudnn-archive

:: 下载 CUDA
echo Downloading CUDA...
powershell -Command "Invoke-WebRequest -Uri %CUDA_INSTALLER% -OutFile cuda_installer.exe"

:: 安装 CUDA
echo Installing CUDA...
start /wait cuda_installer.exe /S

:: 提示用户下载 cuDNN
echo Please download cuDNN from %CUDNN_INSTALLER% and follow the installation instructions.
echo After downloading, extract the files and copy them to the CUDA installation directory.

:: 清理
del cuda_installer.exe
echo CUDA installation complete. Please install cuDNN manually.

pause
