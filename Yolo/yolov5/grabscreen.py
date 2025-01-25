import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api

class ScreenGrabber:
    def __init__(self, use_capture_device=False, device_index=0):
        self.use_capture_device = use_capture_device
        self.device_index = device_index
        # self.device_fps = device_fps

        if self.use_capture_device:
            self.cap = cv2.VideoCapture(self.device_index)
            if not self.cap.isOpened():
                raise ValueError(f"Unable to open capture device {self.device_index}")

    def grab_screen(self, region=None):
        if self.use_capture_device:
            # 设备捕获（例如摄像头）
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to grab frame from capture device")
            height, width, _ = frame.shape
            crop_size = 320
            center_x, center_y = width // 2, height // 2
            left = center_x - crop_size // 2
            top = center_y - crop_size // 2
            right = center_x + crop_size // 2
            bottom = center_y + crop_size // 2
            frame = frame[top:bottom, left:right]
            return frame

        else:
            # 屏幕捕获
            hwin = win32gui.GetDesktopWindow()
            if region:
                left, top, x2, y2 = region
                width = x2 - left
                height = y2 - top
            else:
                # 如果没有提供区域，则捕获整个虚拟屏幕
                width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
                height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
                left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
                top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

            hwindc = win32gui.GetWindowDC(hwin)
            srcdc = win32ui.CreateDCFromHandle(hwindc)
            memdc = srcdc.CreateCompatibleDC()
            bmp = win32ui.CreateBitmap()
            bmp.CreateCompatibleBitmap(srcdc, width, height)
            memdc.SelectObject(bmp)
            memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
            signedIntsArray = bmp.GetBitmapBits(True)
            img = np.frombuffer(signedIntsArray, dtype='uint8')
            img.shape = (height, width, 4)

            srcdc.DeleteDC()
            memdc.DeleteDC()
            win32gui.ReleaseDC(hwin, hwindc)
            win32gui.DeleteObject(bmp.GetHandle())
            return img

    def release(self):
        """
        释放资源
        """
        if self.use_capture_device:
            self.cap.release()
        # cv2.destroyAllWindows()
