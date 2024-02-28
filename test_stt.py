import speech_recognition as sr

def continuous_recognition():
    recognizer = sr.Recognizer()

    # 使用麦克风作为源不断监听
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("现在开始持续监听，请说些什么...")

        while True:  # 无限循环以持续监听
            try:
                print("正在监听...")
                audio = recognizer.listen(source)
                print("识别中...")
                text = recognizer.recognize_google(audio, language='zh-TW')
                print("Google 语音识别认为你说的是： " + text)
            except sr.UnknownValueError:
                print("Google 语音识别无法理解音频")
            except sr.RequestError as e:
                print("从 Google 语音识别服务请求数据失败; {0}".format(e))
            except KeyboardInterrupt:
                print("程序已手动中断")
                break

if __name__ == "__main__":
    continuous_recognition()
