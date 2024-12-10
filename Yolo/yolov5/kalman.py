from filterpy.kalman import KalmanFilter
import numpy as np


class KalmanFilterWrapper:
    def __init__(self, dt, process_noise, measurement_noise, initial_estimate, initial_covariance):
        """
        使用 filterpy 库的 KalmanFilter 替代自定义实现

        :param dt: 时间步长
        :param process_noise: 过程噪声（Q）
        :param measurement_noise: 测量噪声（R）
        :param initial_estimate: 初始状态估计 [位置, 速度]
        :param initial_covariance: 初始协方差矩阵 (P)
        """
        # 创建 KalmanFilter 对象
        self.kf = KalmanFilter(dim_x=2, dim_z=1)

        # 设置状态转移矩阵 (F)，假设只有位置和速度
        self.kf.F = np.array([[1, dt], [0, 1]])  # 状态转移矩阵

        # 设置测量矩阵 (H)，假设你只测量位置
        self.kf.H = np.array([[1, 0]])  # 测量矩阵

        # 过程噪声协方差矩阵 (Q)
        self.kf.Q = np.array([[process_noise, 0], [0, process_noise]])  # 过程噪声

        # 测量噪声协方差矩阵 (R)
        self.kf.R = np.array([[measurement_noise]])  # 测量噪声

        # 初始协方差矩阵 (P)
        self.kf.P = initial_covariance

        # 初始状态估计
        self.kf.x = initial_estimate  # 初始状态估计 [位置, 速度]

    def predict(self):
        """
        使用 filterpy 库进行预测
        """
        self.kf.predict()  # 调用 filterpy 的 predict 方法
        return self.kf.x

    def update(self, measurement):
        """
        使用 filterpy 库进行更新

        :param measurement: 新的测量值（位置）
        """
        self.kf.update(measurement)  # 调用 filterpy 的 update 方法
        return self.kf.x
