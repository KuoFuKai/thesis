import pyttsx3
import time
from threading import Thread
import multiprocessing

# 全域變量，用於控制執行緒的運行
running = True


def threaded(fn):
    def wrapper(*args, **kwargs):
        global running
        running = True  # 每次呼叫執行緒前確保running為True
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper


def speak(phrase):
    engine = pyttsx3.init()
    engine.say(phrase)
    engine.runAndWait()


@threaded
def manage_process(p):
    while p.is_alive() and running:  # 檢查程序是否活著且執行標誌為真
        continue
    if not running:  # 如果執行標誌變成假，則終止進程
        p.terminate()


def say(phrase):
    print(phrase)
    global running
    running = False  # 設定運行標誌為假，以終止先前的執行緒
    time.sleep(0.1)  # 稍作延遲，確保執行緒有時間回應運行標誌的變化

    p = multiprocessing.Process(target=speak, args=(phrase,))
    p.start()
    manage_process(p)


if __name__ == "__main__":
    say("Hello, this is the first message.")
    time.sleep(2)
    say("And this is a second message, after stopping the first one.")
