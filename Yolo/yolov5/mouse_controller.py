# mouse_controller.py

import ctypes
import serial
import time
import win32api
import win32con

# driver_path = r"C:\Users\home123\cq\LGMC\logitech.driver.dll"
driver_path = r"C:\Users\Administrator\PycharmProjects\cq\LGMC\logitech.driver.dll"

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


class CH9350Mouse:
    """
    A class for controlling the mouse using the CH9350 chip.
    This class communicates with CH9350 via a serial interface.
    """

    def __init__(self, port, baudrate=115200):
        """
        Initialize the CH9350 mouse controller.
        :param port: Serial port (e.g., "COM3")
        :param baudrate: Baud rate for serial communication
        """
        try:
            self.ser = serial.Serial(port, baudrate)
            # time.sleep(2)  # Wait for the serial connection to initialize
            print(f"Connected to CH9350 on {port} with baudrate {baudrate}")
        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            self.ser = None

    def move(self, x, y):
        """
        Move the mouse relative to its current position using CH9350.
        :param x: Horizontal movement, positive is to the right, negative is to the left
        :param y: Vertical movement, positive is downward, negative is upward
        """
        if not self.ser:
            print("Serial connection not initialized")
            return
        # Ensure x and y are within CH9350's accepted range (-127 to 127)
        x = max(-127, min(127, x))
        y = max(-127, min(127, y))

        # Build the message following the format 57 AB 02 00 0A 05 00
        # Here 57, AB are likely fixed header bytes, 02 could be the command type (e.g., "move"),
        # followed by parameters for movement, and the final byte could be some sort of checksum or end byte.

        command = bytearray([0x57, 0xAB, 0x02, 0x00, x & 0xFF, y & 0xFF, 0x00])

        # Send the command to CH9350
        self.ser.write(command)

    def click(self, button=1):
        """
        Simulate a mouse click using CH9350.
        :param button: 1 for left-click, 2 for right-click
        """
        if not self.ser:
            print("Serial connection not initialized")
            return
        # Define command format for click (similar to the move function)
        if button == 1:
            # Left-click: 57 AB 03 00 01 00
            command = bytearray([0x57, 0xAB, 0x03, 0x00, 0x01, 0x00, 0x00])
        elif button == 2:
            # Right-click: 57 AB 03 00 02 00
            command = bytearray([0x57, 0xAB, 0x03, 0x00, 0x02, 0x00, 0x00])

        # Send the click command to CH9350
        self.ser.write(command)

    def close(self):
        """
        Close the serial connection.
        """
        if self.ser:
            self.ser.close()
            print("Serial connection closed")


# Usage Guide:
# -------------------------------------------
# Example for using CH9350Mouse:
#
# from mouse_controller import CH9350Mouse
#
# mouse = CH9350Mouse(port="COM3")  # Initialize the CH9350 mouse on COM3
# mouse.move(50, 20)               # Move the mouse 50 pixels to the right, 20 pixels down
# mouse.click(1)                   # Left-click
# mouse.close()                    # Close the serial connection
#
# -------------------------------------------
# Notes:
# - Ensure the CH9350 chip is connected to the specified COM port.
# - Adjust the baud rate if necessary to match CH9350's default settings.
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
