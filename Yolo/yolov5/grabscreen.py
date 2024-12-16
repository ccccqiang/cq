import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api

class ScreenGrabber:
    def __init__(self, use_capture_device=False, device_index=0):
        self.use_capture_device = use_capture_device
        self.device_index = device_index
        if self.use_capture_device:
            self.cap = cv2.VideoCapture(self.device_index)
            if not self.cap.isOpened():
                raise ValueError(f"Unable to open capture device {self.device_index}")

    def grab_screen(self, region=None):
        if self.use_capture_device:
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to grab frame from capture device")
            if region:
                # 裁剪到指定区域
                left, top, right, bottom = region
                width, height = right - left, bottom - top
                print("left:", left, "top:", top, "width:", width, "height:", height)
                frame = frame[top:bottom, left:right]
            cv2.imshow('Image', frame)
            cv2.waitKey(0)  # 按任意键关闭窗口
            return frame

        # 屏幕抓取逻辑
        # ...（省略其他代码）


        # 屏幕抓取逻辑
        hwin = win32gui.GetDesktopWindow()
        if region:
            left, top, x2, y2 = region
            width = x2 - left
            height = y2 - top
            print("left:", left, "top:", top, "width:", width, "height:", height)
        else:
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
        cv2.imshow('Image', img)
        cv2.waitKey(0)  # 按任意键关闭窗口
        return img

    def release(self):
        """
        释放资源
        """
        if self.use_capture_device:
            self.cap.release()
        cv2.destroyAllWindows()
