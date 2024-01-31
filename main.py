# main.py
import threading
import pyttsx3
import speech_recognition as sr
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from llm_setup import llm_setup
from yolo_inference import yolo_inference, detected_queue
from queue import Empty

# 初始化 LLM
llm = llm_setup("MediaTek-Research/Breeze-7B-Instruct-64k-v0_1")

# 定義 prompt 模板
template = """<s>[INST] 你是導遊，遇到一個物體，負責講解該物體和解答該物體相關的問題
。 物體：{object}，問題：{question} [/INST] </s> """
prompt = PromptTemplate(template=template, input_variables=["question", "object"])

# 初始化 TTS 引擎
engine = pyttsx3.init()

# 初始化語音辨識引擎
recognizer = sr.Recognizer()

# 執行物件辨識
threading.Thread(target=yolo_inference, args=("test.mp4",), daemon=True).start()
detected_obj = None  # 現在的對象
last_detected_obj = None  # 用於記錄最後一次偵測到的對象


def ask_question(question):
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.run({"question": question, "object": detected_obj})
    return response


# 主循環
while True:
    try:
        obj = detected_queue.get_nowait()  # 預防阻塞用get_nowait, 而非get
    except Empty:
        obj = None  # 如果隊列為空，則保持obj為None

    detected_obj = obj if obj else last_detected_obj
    if detected_obj and not last_detected_obj:  # 第一次偵測到東西
        print("第一次偵測到東西")
        answer = ask_question("現在出現在面前的新物體，請簡短快速簡介")
        print(answer)
        engine.say(answer)
        engine.runAndWait()
        last_detected_obj = detected_obj
    elif detected_obj != last_detected_obj:  # 偵測到新東西 (非第一次)
        print("偵測到新東西 (非第一次)")
        answer = ask_question("現在出現在面前的新物體，請簡短快速簡介")
        print(answer)
        engine.say(answer)
        engine.runAndWait()
        last_detected_obj = detected_obj

    with sr.Microphone() as source:
        print("請問您的問題（或輸入 'exit' 退出）: ")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            user_input = recognizer.recognize_google(audio, language='zh-TW')
            if user_input.lower() == 'exit':
                break

            print(f"您問的問題是: '{user_input}', 是否繼續？")
            engine.say(f"您問的問題是: '{user_input}', 是否繼續？")
            engine.runAndWait()

            with sr.Microphone() as source2:
                recognizer.adjust_for_ambient_noise(source2)
                audio = recognizer.listen(source2)
                try:
                    confirmation = recognizer.recognize_google(audio, language='zh-TW')
                    print(confirmation)
                except sr.UnknownValueError:
                    print("Google Speech Recognition 無法理解音訊")
                except sr.RequestError:
                    print("無法從 Google Speech Recognition 服務請求資料")

            if '繼續' in confirmation:
                answer = ask_question(user_input)
                print(answer)
                engine.say(answer)
                engine.runAndWait()
            else:
                print("已取消")
        except sr.UnknownValueError:
            print("Google Speech Recognition 無法理解音訊")
        except sr.RequestError:
            print("無法從 Google Speech Recognition 服務請求資料")