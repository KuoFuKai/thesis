import threading
from ultralytics import YOLO  # 引入 YOLO 模型
from llm_setup import llm_setup
from llm_util import interact
from yolo_util import inference
from tts_util import say

if __name__ == '__main__':
    say("程式初始化請稍後")
    # 初始化 LLM
    llm = llm_setup("MediaTek-Research/Breeze-7B-Instruct-v0_1")
    say("載入大語言模型成功")
    # 初始化 Yolo
    model = YOLO("best.pt")
    say("載入物件辨識模型成功")
    # 執行物件辨識
    threading.Thread(target=inference, args=("test.mp4", model, llm,)).start()
    # 進行QA環節
    threading.Thread(target=interact, args=(llm,)).start()
