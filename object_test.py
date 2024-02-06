from llm_util import ask_question
import threading


class TestThread(threading.Thread):
    def __init__(self, llm, yolo_tts_engine):
        threading.Thread.__init__(self)
        self.llm = llm
        self.yolo_tts_engine = yolo_tts_engine
        self._stop_event = threading.Event()
        self._is_running = False

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        self._is_running = True
        while not self.stopped():
            answer = ask_question(self.llm, "現在出現在面前的新物體，請簡短快速簡介")
            print(answer)
            self.yolo_tts_engine.say(answer)
            self.yolo_tts_engine.runAndWait()
        self._is_running = False

    def stop(self):
        if self.is_running():
            self._stop_event.set()
            # self.yolo_tts_engine.stop()
            self.yolo_tts_engine.endLoop()
            self.yolo_tts_engine.stop()

    def is_running(self):
        return self._is_running

    def reset(self):
        self._stop_event.clear()

    def restart(self):
        if not self.is_running():
            self.reset()
            self.start()
