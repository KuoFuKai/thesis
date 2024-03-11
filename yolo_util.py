# yolo_util.py
import sys
import cv2  # 引入 OpenCV 庫
import math  # 引入數學庫用于計算
import variable
from queue import Queue
from tts_util import say
from llm_util import ask_question
from csi_camera import gstreamer_pipeline

detected_queue = Queue()
conf_threshold = 0.8


def inference(source, model, llm, rag):
    # classNames = ["老虎", "小老虎", "白老虎"]
    classNames = ["全臺首學", "台南孔廟明倫堂", "台南孔廟文昌閣", "台南孔廟泮宮坊"]
    max_confidence = 0  # 最大信心值
    last_object = None  # 最新物件
    paused = False  # 暫停狀態

    # 確定視頻來源是攝像頭還是視頻文件
    if source == "CSI":
        cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    elif source == "webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)

    if source not in "CSI":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 設置寬度
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 設置高度

    if not cap.isOpened():
        print("無法打開相機")
        sys.exit()
        return

    while True:
        if not paused and not variable.pause_detect_event.is_set():
            success, img = cap.read()
            if not success:
                break  # 視頻結束或攝像頭關閉時退出循環

            results = model(img, stream=True, verbose=False)  # verbose=False關閉輸出結果到console

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    confidence = math.ceil(box.conf[0] * 100) / 100
                    # print(cls, confidence)  #調整信心值參考
                    if confidence >= conf_threshold:
                        # x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        # cv2.putText(img, f"{classNames[cls]} {confidence:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        #             (255, 0, 0), 2)
                        if confidence > max_confidence or (len(boxes) == 1 and confidence >= conf_threshold):
                            max_confidence = confidence
                            detected_obj = classNames[cls]
                            if last_object != detected_obj:
                                print(detected_obj)
                                last_object = detected_obj
                                variable.detected_obj = detected_obj

                                say("現在偵測到新物體"+detected_obj+"準備為您介紹")
                                answer = ask_question(llm, rag, "請簡短快速簡介")
                                variable.pause_detect_event.set()
                                say(answer)

        # cv2.imshow('YOLO Inference', img)

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
    from llm_setup import llm_setup, rag_setup

    # 初始化 LLM
    llm = llm_setup("MediaTek-Research/Breeze-7B-Instruct-64k-v0_1")
    rag = rag_setup()
    # 初始化 Yolo
    model = YOLO("best_tainan.pt")

    inference("webcam", model, llm, rag, )

