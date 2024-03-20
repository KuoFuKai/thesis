import argparse
import multiprocessing
import threading
from ultralytics import YOLO  # 引入 YOLO 模型
from llm_setup import tokenizer_setup, streamer_setup, llm_setup, rag_setup
from llm_util import interact
from yolo_util import inference
from tts_util import say
from log_util import start_logging_thread

# 設定命令行參數
parser = argparse.ArgumentParser(description="啟動物件辨識和語言互動系統")
parser.add_argument("--source", type=str, default="CSI", help="指定影像來源，可以是 'CSI'、'webcam' 或影片的路徑")
args = parser.parse_args()

if __name__ == '__main__':
    say("程式初始化請稍後")
    # 初始化 LLM
    llm_model = "MediaTek-Research/Breeze-7B-Instruct-v0_1"
    tokenizer = tokenizer_setup(llm_model)
    streamer = streamer_setup(tokenizer)
    llm = llm_setup(llm_model, tokenizer, streamer)
    rag = rag_setup()
    say("載入大語言模型成功")
    # 初始化 Yolo
    yolo_model = YOLO("best_tainan.pt")
    say("載入物件辨識模型成功")
    # 設置Multi-Processing方式 (For Linux)
    multiprocessing.set_start_method('spawn')
    # 開始記錄LOG
    start_logging_thread()
    # 執行物件辨識
    threading.Thread(target=inference, args=(args.source, yolo_model, llm, rag, streamer, )).start()
    # 進行QA環節
    threading.Thread(target=interact, args=(llm, rag, streamer, )).start()
