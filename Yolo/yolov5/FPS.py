import time

class FPS:
    def __init__(self):
        self.prev_time = time.time()  # 上一帧的时间
        self.frame_count = 0  # 帧计数
        self.fps = 0  # 每秒帧数

    def update(self):
        """ 更新帧数并计算FPS """
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.prev_time >= 1.0:  # 每秒更新一次
            self.fps = self.frame_count / (current_time - self.prev_time)
            self.frame_count = 0  # 重置帧计数
            self.prev_time = current_time  # 重置时间
            print(f"FPS: {self.fps:.2f}")  # 打印当前FPS
