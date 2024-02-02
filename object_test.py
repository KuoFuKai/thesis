from llm_util import ask_question


def test(llm, yolo_tts_engine):
    answer = ask_question(llm, "現在出現在面前的新物體，請簡短快速簡介")
    print(answer)
    yolo_tts_engine.say(answer)
    yolo_tts_engine.runAndWait()
