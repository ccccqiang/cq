import pyautogui

def move_mouse_to_head(coordinates):
    # 假设coordinates的格式为 (x, y, class_id, confidence)
    # 仅移动到 CT Head (ID 1) 和 T Head (ID 3) 的坐标，并且置信度大于0.95
    head_coordinates = [(x, y) for (x, y, class_id, confidence) in coordinates if class_id in [1, 3] and confidence > 0.95]

    for (x, y) in head_coordinates:
        pyautogui.moveTo(x, y, duration=0.1)  # 移动鼠标
