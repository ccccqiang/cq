import numpy as np


class KalmanFilter:
    def __init__(self, dt, process_noise, measurement_noise, initial_estimate, initial_covariance):
        """
        Kalman Filter initialization

        :param dt: Time step
        :param process_noise: Process noise (Q)
        :param measurement_noise: Measurement noise (R)
        :param initial_estimate: Initial state estimate [position, velocity]
        :param initial_covariance: Initial covariance matrix (P)
        """
        self.dt = dt  # Time step
        self.F = np.array([[1, dt], [0, 1]])  # State transition matrix
        self.H = np.array([[1, 0]])  # Measurement matrix
        self.Q = np.array([[process_noise, 0], [0, process_noise]])  # Process noise covariance
        self.R = np.array([[measurement_noise]])  # Measurement noise covariance
        self.P = initial_covariance  # Initial covariance
        self.x = initial_estimate  # Initial state estimate

    def predict(self):
        """
        Predict the next state (position and velocity) based on the model.
        """
        self.x = np.dot(self.F, self.x)  # State prediction
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q  # Covariance prediction
        return self.x

    def update(self, measurement):
        """
        Update the state estimate based on the new measurement.

        :param measurement: New measurement (position)
        """
        y = measurement - np.dot(self.H, self.x)  # Measurement residual
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # Residual covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman gain
        self.x = self.x + np.dot(K, y)  # State update
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)  # Covariance update
        return self.x
