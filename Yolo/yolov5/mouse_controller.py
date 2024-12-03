# mouse_controller.py

import ctypes
import win32api
import win32con

driver_path = r"C:\\Users\\home123\\cq\\LGMC\\logitech.driver.dll"

class LogitechMouse:
    """
    A class for controlling the Logitech mouse using the Logitech driver.
    """
    def __init__(self, driver_path=driver_path):
        try:
            # Load the Logitech driver DLL
            self.driver = ctypes.CDLL(driver_path)
            self.ok = self.driver.device_open() == 1  # The driver can only be opened once per process
            if not self.ok:
                print('Error, GHUB or LGS driver not found')
        except FileNotFoundError:
            print(f'Error, DLL file not found')

    def press(self, code):
        if not self.ok:
            return
        self.driver.mouse_down(code)

    def release(self, code):
        if not self.ok:
            return
        self.driver.mouse_up(code)

    def click(self, code):
        if not self.ok:
            return
        self.driver.mouse_down(code)
        self.driver.mouse_up(code)

    def scroll(self, a):
        if not self.ok:
            return
        self.driver.scroll(a)

    def move(self, x, y):
        """
        Move the mouse relative to its current position.
        :param x: Horizontal movement, positive is to the right, negative is to the left
        :param y: Vertical movement, positive is downward, negative is upward
        """
        if not self.ok:
            return
        if x == 0 and y == 0:
            return
        self.driver.moveR(x, y, True)  # Relative movement


class LogitechKeyboard:
    """
    A class for controlling the keyboard using the Logitech driver.
    The `code` can be any key code like 'a'-'z', '0'-'9', etc.
    """
    def __init__(self, driver_path=driver_path):
        try:
            # Load the Logitech driver DLL
            self.driver = ctypes.CDLL(driver_path)
            self.ok = self.driver.device_open() == 1  # The driver can only be opened once per process
            if not self.ok:
                print('Error, GHUB or LGS driver not found')
        except FileNotFoundError:
            print(f'Error, DLL file not found')

    def press(self, code):
        if not self.ok:
            return
        self.driver.key_down(code)

    def release(self, code):
        if not self.ok:
            return
        self.driver.key_up(code)

    def click(self, code):
        if not self.ok:
            return
        self.driver.key_down(code)
        self.driver.key_up(code)


# Usage Guide:
# -------------------------------------------
# Example for using LogitechMouse:
#
# from mouse_controller import LogitechMouse
#
# mouse = LogitechMouse()  # Initialize the mouse
# mouse.move(100, 50)      # Move the mouse 100 pixels to the right and 50 pixels down
# mouse.click(1)           # Left-click using code 1 (left button)
# mouse.scroll(1)          # Scroll up
#
# -------------------------------------------
# Example for using LogitechKeyboard:
#
# from mouse_controller import LogitechKeyboard
#
# keyboard = LogitechKeyboard()  # Initialize the keyboard
# keyboard.press(ord('A'))       # Press the 'A' key (use ord() to convert 'A' to its ASCII code)
# keyboard.release(ord('A'))     # Release the 'A' key
# keyboard.click(ord('A'))       # Click the 'A' key (press and release)
#
# -------------------------------------------
# Notes:
# - Make sure the Logitech driver is properly installed and available at the specified path.
# - The `driver_path` variable points to the driver DLL file location. Update it as needed.
# - The `ok` attribute indicates whether the driver loaded successfully.
# - Replace `1` with the appropriate mouse button code for middle/right-click actions:
#     - 1: Left button
#     - 2: Right button
#     - 3: Middle button
# - Ensure you have permissions to access the driver, and GHUB or LGS software is installed.
