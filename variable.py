import threading

detected_obj = None  # 現在的對象
last_detected_obj = None  # 用於記錄最後一次偵測到的對象
pause_detect_event = threading.Event()  # 控制物件辨識執行緒
pause_interact_event = threading.Event()  # 控制互動執行緒
