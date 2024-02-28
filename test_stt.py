import speech_recognition as sr

def test_microphone():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("请在麦克风前说些什么，我会尝试识别...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        print("正在识别...")

        try:
            text = recognizer.recognize_google(audio, language='zh-TW')
            print("Google 语音识别认为你说的是： " + text)
        except sr.UnknownValueError:
            print("Google 语音识别无法理解音频")
        except sr.RequestError as e:
            print("从 Google 语音识别服务请求数据失败; {0}".format(e))

if __name__ == "__main__":
    test_microphone()