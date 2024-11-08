import bettercam
import cv2
import time

# 创建 BetterCam 实例
camera = bettercam.create()

# 设置捕获区域为 320x320，屏幕中间
screen_width = 1920  # 屏幕宽度
screen_height = 1080  # 屏幕高度
left, top = (screen_width - 320) // 2, (screen_height - 320) // 2
right, bottom = left + 320, top + 320
region = (left, top, right, bottom)

# 启动屏幕捕获并指定区域
camera.start(region=region)

# FPS 计算变量
prev_time = time.time()
fps_display_interval = 1  # 每秒显示一次帧率
fps_counter = 0  # 计数帧数
fps = 0  # 最终显示的帧率

# 捕获并实时显示
while True:
    start_time = time.time()
    frame = camera.get_latest_frame()  # 获取最新的帧

    # 将帧数据从 RGB 转换为 BGR（OpenCV 需要 BGR 格式）
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 显示捕获的区域
    cv2.imshow("Captured Region", frame_bgr)
    # 计算当前帧的时间
    frame_time = time.time() - start_time
    # 计算并显示 FPS
    fps_counter += 1
    if time.time() - prev_time >= fps_display_interval:
        fps = fps_counter / fps_display_interval
        fps_counter = 0
        prev_time = time.time()
    # 在终端输出帧数和每帧处理时间
    print(f"Frame Time: {frame_time * 1000:.2f} ms, FPS: {fps:.2f}")
    # 等待 1 毫秒，按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 停止捕获并释放资源
camera.stop()
cv2.destroyAllWindows()
