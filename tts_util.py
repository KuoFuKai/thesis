import pyttsx3
import time
from threading import Thread
import multiprocessing

# 全局变量，用于控制线程的运行
running = True

def threaded(fn):
    def wrapper(*args, **kwargs):
        global running
        running = True  # 每次调用线程前确保running为True
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
    while p.is_alive() and running:  # 检查进程是否活着且运行标志为真
        continue
    if not running:  # 如果运行标志变为假，则终止进程
        p.terminate()

def say(phrase):
    global running
    running = False  # 设置运行标志为假，以终止之前的线程
    time.sleep(0.1)  # 稍作延迟，确保线程有时间响应运行标志的变化

    p = multiprocessing.Process(target=speak, args=(phrase,))
    p.start()
    manage_process(p)

if __name__ == "__main__":
    say("Hello, this is the first message.")
    time.sleep(2)
    say("And this is a second message, after stopping the first one.")
