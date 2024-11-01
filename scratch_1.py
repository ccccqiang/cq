import cv2
import numpy as np
import onnxruntime as ort
import mss
import time


class Model:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_shape[3], self.input_shape[2]))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img

    def detect(self, img):
        output = self.session.run([self.output_name], {self.input_name: img})
        return output[0]


def capture_screen(region):
    with mss.mss() as sct:
        img = sct.grab(region)
        img_np = np.array(img)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
        return img_np


def draw_boxes(frame, detections, conf_threshold=0.5):
    for detection in detections:
        if len(detection) < 5:  # 确保检测数据完整
            continue
        x, y, w, h, confidence = detection
        if confidence >= conf_threshold:
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            cv2.putText(frame, f'Conf: {confidence:.2f}', (int(x), int(y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def main_loop(model):
    imgsize = model.input_shape[2]
    region = {"top": (1080 - imgsize) // 2, "left": (1920 - imgsize) // 2, "width": imgsize, "height": imgsize}

    while True:
        try:
            frame = capture_screen(region)
            input_data = model.preprocess(frame)
            output = model.detect(input_data)

            detections = output[0]
            boxes = []
            for detection in detections:
                box = detection[:5]
                boxes.append(box)

            draw_boxes(frame, boxes)

            cv2.imshow("Output", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # time.sleep(0.03)  # 控制帧率

        except Exception as e:
            print(f"Error: {e}")
            break


if __name__ == "__main__":
    model_path = "E:\\123pan\\Downloads\\ai\\onnx\\cs2.onnx"
    model = Model(model_path)

    main_loop(model)

    cv2.destroyAllWindows()
