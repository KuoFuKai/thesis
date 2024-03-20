import multiprocessing
import pyttsx3

import time

current_voice_process = None


def speak(phrase):
    engine = pyttsx3.init()
    engine.setProperty('voice', 'zh')
    engine.say(phrase)
    engine.runAndWait()


def terminate_current_voice_process():
    global current_voice_process
    if current_voice_process and current_voice_process.is_alive():
        current_voice_process.terminate()
        current_voice_process.join()  # 確保進程已完全結束
        current_voice_process = None


def say(phrase):
    print(phrase)
    global current_voice_process
    terminate_current_voice_process()  # 結束目前正在執行的語音輸出進程
    # 建立並啟動新的語音輸出進程
    current_voice_process = multiprocessing.Process(target=speak, args=(phrase, ))
    current_voice_process.start()


def queue_worker(phrases):
    engine = pyttsx3.init()
    engine.setProperty('voice', 'zh')
    while True:
        phrase = phrases.get()  # 从队列中获取一个项目
        if phrase is None:  # 检查是否为哨兵值，用于结束循环
            break
        engine.say(phrase)
        engine.runAndWait()


def say_queue(phrases):
    global current_voice_process
    terminate_current_voice_process()  # 結束目前正在執行的語音輸出進程
    # 建立並啟動新的語音輸出進程
    current_voice_process = multiprocessing.Process(target=queue_worker, args=(phrases, ))
    current_voice_process.start()


if __name__ == "__main__":
    # 使用say方法
    say("Starting the voice output.")
    time.sleep(1)  # 等待一些時間讓say方法有機會執行
    # 使用say_queue方法，預期會終止say方法的進程
    say_queue(["This is a message from say_queue.", "This is a message from say_queue.", "This is a message from say_queue.", "This is a message from say_queue."])
    time.sleep(1)  # 等待一些時間讓say_queue有機會執行
    # 再次使用say方法，預期會終止say_queue的進程
    say("Another message from say, after say_queue.")
    time.sleep(1)  # 等待一些時間讓say_queue有機會執行
    say_queue(["This is a message from say_queue.", "This is a message from say_queue."])
    time.sleep(1)  # 等待一些時間讓say_queue有機會執行
    say_queue(["This is last.", None])
