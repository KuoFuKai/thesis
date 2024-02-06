import multiprocessing
import pyttsx3
import time
from threading import Thread


# 使用裝飾器來使函數在新的線程中執行
def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper


# 语音播放函数
def speak(phrase):
    engine = pyttsx3.init()  # 初始化語音引擎
    engine.say(phrase)  # 播放傳入的語句
    engine.runAndWait()  # 等待播放完成
    engine.stop()  # 停止引擎


# 停止語音播放的函數
def stop_speaker():
    global term
    term = True  # 設置終止標誌為True
    t.join()  # 等待管理進程的線程結束


# 管理進程的函數，用於監控並在需要時停止進程
@threaded
def manage_process(p):
    global term
    while p.is_alive():  # 檢查進程是否仍在運行
        if term:  # 如果終止標誌被設置為True
            p.terminate()  # 終止進程
            term = False  # 重置終止標誌
        else:
            continue


# 主要語音播放函數，啟動一個進程來播放語音，並用一個線程來管理這個進程
def say(phrase):
    global t
    global term
    term = False  # 重置終止標誌
    p = multiprocessing.Process(target=speak, args=(phrase,))  # 創建一個進程來播放語音
    p.start()  # 啟動進程
    t = manage_process(p)  # 啟動一個線程來管理進程


if __name__ == "__main__":
    say("this process is running right now")  # 開始播放語音
    time.sleep(1)  # 暫停1秒
    stop_speaker()  # 停止播放
    say("this process is running right now")  # 再次開始播放
    time.sleep(1.5)  # 暫停1.5秒
    stop_speaker()  # 再次停止播放