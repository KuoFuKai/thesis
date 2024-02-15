# yolo_util.py
import time

import cv2  # 引入 OpenCV 庫
import math  # 引入數學庫用于計算
import variable
from queue import Queue
from llm_util import ask_question
from tts_util import say

detected_queue = Queue()
conf_threshold = 0.8


def inference(source, model, llm):
    classNames = ["老虎", "小老虎", "白老虎"]
    max_confidence = 0  # 最大信心值
    last_object = None  # 最新物件
    paused = False  # 暫停狀態

    # 確定視頻來源是攝像頭還是視頻文件
    if source == "webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)

    cap.set(3, 640)  # 設置寬度
    cap.set(4, 480)  # 設置高度

    while True:
        if not paused:
            success, img = cap.read()
            if not success:
                break  # 視頻結束或攝像頭關閉時退出循環

            results = model(img, stream=True, verbose=False)  # verbose=False關閉輸出結果到console

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
                            detected_obj = classNames[cls]
                            if last_object != detected_obj:
                                print(detected_obj)
                                last_object = detected_obj
                                variable.detected_obj = detected_obj

                                say("現在偵測到新物體"+detected_obj+"準備為您介紹")
                                time.sleep(5)
                                answer = ask_question(llm, "現在出現在面前的新物體，請簡短快速簡介")
                                say(answer)

        cv2.imshow('YOLO Inference', img)

        # 按鍵處理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 按 'q' 退出
            break
        elif key == ord(' '):  # 按空格鍵暫停/繼續
            paused = not paused

    # 釋放資源並關閉窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    from ultralytics import YOLO  # 引入 YOLO 模型
    from llm_setup import llm_setup

    # 初始化 LLM
    llm = llm_setup("MediaTek-Research/Breeze-7B-Instruct-64k-v0_1")
    # 初始化 Yolo
    model = YOLO("best.pt")

    inference("test.mp4", model, llm, )
