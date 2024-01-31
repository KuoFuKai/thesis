# yolo_inference.py
from ultralytics import YOLO  # 引入 YOLO 模型
import cv2  # 引入 OpenCV 庫
import math  # 引入數學庫用于計算
from queue import Queue

detected_queue = Queue()
conf_threshold = 0.8


def yolo_inference(source):
    # 确定视频来源是摄像头还是视频文件
    if source == "webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)

    cap.set(3, 640)  # 设置宽度
    cap.set(4, 480)  # 设置高度

    # 加载 YOLO 模型
    model = YOLO("best.pt")
    classNames = ["tiger", "tiger-cub", "white-tiger"]

    max_confidence = 0

    paused = False  # 暂停状态

    while True:
        if not paused:
            success, img = cap.read()
            if not success:
                break  # 视频结束或摄像头关闭时退出循环

            results = model(img, stream=True)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    confidence = math.ceil(box.conf[0] * 100) / 100
                    if confidence >= conf_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        cv2.putText(img, f"{classNames[cls]} {confidence:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 0), 2)
                        if confidence > max_confidence or (len(boxes) == 1 and confidence >= conf_threshold):
                            max_confidence = confidence
                            detected_queue.put(classNames[cls])

        cv2.imshow('YOLO Inference', img)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 按 'q' 退出
            break
        elif key == ord(' '):  # 按空格键暂停/继续
            paused = not paused

    # 釋放資源幷關閉窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    yolo_inference("test.mp4")
