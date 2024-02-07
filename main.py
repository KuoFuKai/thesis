# main.py0
import threading
import pyttsx3
from ultralytics import YOLO  # 引入 YOLO 模型
from llm_setup import llm_setup
from llm_util import interact
from yolo_util import inference

if __name__ == '__main__':
    # 初始化 LLM
    llm = llm_setup("MediaTek-Research/Breeze-7B-Instruct-64k-v0_1")
    # 初始化 Yolo
    model = YOLO("best.pt")
    # 執行物件辨識
    threading.Thread(target=inference, args=("test.mp4", model, llm,)).start()
    # 進行QA環節
    threading.Thread(target=interact, args=(llm,)).start()
