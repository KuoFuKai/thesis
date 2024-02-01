from yolo_inference import detected_queue
from queue import Empty
from llm_util import ask_question
import pyttsx3
import variable


def new_object(llm):
    # 初始化 TTS 引擎
    engine = pyttsx3.init()
    while True:
        try:
            obj = detected_queue.get_nowait()  # 預防阻塞用get_nowait, 而非get
        except Empty:
            obj = None  # 如果隊列為空，則保持obj為None

        variable.detected_obj = obj if obj else variable.last_detected_obj
        if variable.detected_obj and not variable.last_detected_obj:  # 第一次偵測到東西
            print("第一次偵測到東西")
            answer = ask_question(llm, "現在出現在面前的新物體，請簡短快速簡介")
            print(answer)
            engine.say(answer)
            engine.runAndWait()
            variable.last_detected_obj = variable.detected_obj
        elif variable.detected_obj != variable.last_detected_obj:  # 偵測到新東西 (非第一次)
            print("偵測到新東西 (非第一次)")
            answer = ask_question(llm, "現在出現在面前的新物體，請簡短快速簡介")
            print(answer)
            engine.say(answer)
            engine.runAndWait()
            variable.last_detected_obj = variable.detected_obj
