import sys
import os
import threading
import queue
from datetime import datetime

log_queue = queue.Queue()

# 保存原始的stdout和stderr
original_stdout = sys.stdout
original_stderr = sys.stderr


class ThreadSafeLogger(object):
    def __init__(self, queue, original):
        self.queue = queue
        self.original = original

    def write(self, msg):
        # 将消息发送到原始stdout或stderr，并确保自动换行的行为不变
        self.original.write(msg)
        self.original.flush()  # 确保即时输出到控制台
        if msg.strip() != "":  # 避免将空消息或仅换行符的消息写入日志
            self.queue.put(msg)

    def flush(self):
        self.original.flush()


def log_worker(log_filename):
    with open(log_filename, "a", encoding='utf-8') as f:
        while True:
            record = log_queue.get()
            # 确保记录以换行符结束
            if not record.endswith('\n'):
                record += '\n'
            f.write(record)
            f.flush()
            log_queue.task_done()


def start_logging_thread():
    # 重定向stdout和stderr，同时保留控制台输出并确保自动换行
    sys.stdout = ThreadSafeLogger(log_queue, original_stdout)
    sys.stderr = ThreadSafeLogger(log_queue, original_stderr)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f"app_log_{current_time}.log")
    thread = threading.Thread(target=log_worker, args=(log_filename,))
    thread.start()
    return thread


if __name__ == "__main__":
    log_thread = start_logging_thread()
    try:
        print("这是一条通过重定向stdout输出的日志消息")
        # 模拟一个错误输出
        raise Exception("这是一条通过重定向stderr输出的日志消息")
    except Exception as e:
        print(e)
    finally:
        log_queue.put("STOP")
        log_thread.join()
