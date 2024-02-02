# main.py0
import threading
import pyttsx3
from ultralytics import YOLO  # 引入 YOLO 模型
from llm_setup import llm_setup
from llm_util import qa
from yolo_util import inference

if __name__ == '__main__':
    # 初始化 LLM
    llm = llm_setup("MediaTek-Research/Breeze-7B-Instruct-64k-v0_1")
    # 初始化 Yolo
    model = YOLO("best.pt")
    # 初始化 TSS引擎
    yolo_tts_engine = pyttsx3.init()
    llm_tts_engine = pyttsx3.init()
    # 執行物件辨識
    threading.Thread(target=inference, args=("test.mp4", model, llm, yolo_tts_engine,)).start()
    # 進行QA環節
    threading.Thread(target=qa, args=(llm, llm_tts_engine, )).start()
