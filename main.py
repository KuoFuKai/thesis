# main.py0
import threading
from llm_setup import llm_setup
from llm_util import qa
from yolo_inference import yolo_inference, detected_queue
from object_util import new_object

if __name__ == '__main__':
    # 初始化 LLM
    llm = llm_setup("MediaTek-Research/Breeze-7B-Instruct-64k-v0_1")
    # 執行物件辨識
    threading.Thread(target=yolo_inference, args=("test.mp4",)).start()
    # 判斷新物件或舊物件
    threading.Thread(target=new_object, args=(llm, )).start()
    # 進行QA環節
    threading.Thread(target=qa, args=(llm, )).start()
