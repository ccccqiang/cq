import d3dshot
import time
import threading

# 初始化 D3DShot 对象
d = d3dshot.create()

# FPS 计算
frame_count = 0
start_time = time.time()


# 捕获并显示 FPS
def capture_and_display():
    global frame_count, start_time
    try:
        while True:
            # 获取最新帧
            frame = d.get_latest_frame()

            if frame is not None:
                # 每秒更新 FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1.0:
                    fps = frame_count / elapsed_time
                    print(f"FPS: {fps:.2f}")
                    # 重置时间和帧计数
                    start_time = time.time()
                    frame_count = 0

    except KeyboardInterrupt:
        print("捕获已停止。")


# 启动捕获线程
def start_capture(target_fps=144):
    """
    启动屏幕捕获线程，并返回是否成功启动。

    :param target_fps: 每秒捕获的目标帧数
    :return: 布尔值，指示是否成功启动捕获线程
    """
    try:
        # 启动屏幕捕获
        d.capture(target_fps=target_fps)

        # 启动捕获线程
        capture_thread = threading.Thread(target=capture_and_display)
        capture_thread.start()

        print("开始高速屏幕捕获...")
        return True
    except Exception as e:
        print(f"启动捕获线程失败: {e}")
        return False


# 启动捕获并返回结果
capture_success = start_capture(target_fps=144)

if capture_success:
    print("捕获线程成功启动。")
else:
    print("捕获线程启动失败。")

# 等待线程结束
while True:
    try:
        time.sleep(1)  # 休眠，等待捕获线程运行
    except KeyboardInterrupt:
        break

# 停止捕获
d.stop()
print("捕获已停止。")
