import os
import speech_recognition as sr
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import variable
from tts_util import say

# 定義一個prompt模板。
template = """<s>[INST]
你是一名導遊，遊客都只聽得懂中文，眼前看到的建築物，若請你介紹，請你簡短介紹，若問您問題，也請您簡短回答。
提示：{context}
問題：{question}
[/INST] </s> """
# 使用模板和定義的問題與上下文創建一個LLMChain實例。
prompt = PromptTemplate(template=template,
                        input_variables=["context", "question"], )


def ask_question(llm, rag, question):
    say("處理中請稍後")
    formatted_question = "{object}，{question}".format(object=variable.detected_obj, question=question)
    rag_chain = RetrievalQA.from_chain_type(llm=llm,
                                            retriever=rag,
                                            chain_type="stuff",
                                            chain_type_kwargs={"prompt": prompt}, )
    return rag_chain.invoke({"query": formatted_question})["result"]


def interact(llm, rag):
    # 初始化語音辨識引擎
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        while True:
            print("請說出您的問題（或說出 '關機' 退出）: ")
            recognizer.adjust_for_ambient_noise(source, duration=5)  # 減少 duration 以加快反應速度
            audio = recognizer.listen(source)

            try:
                user_input = recognizer.recognize_whisper(audio, language="chinese", model="base")
                if user_input in ['', '字幕by索兰娅', '字幕製作人Zither Harp', 'fashioned视频區', '我看你很像你', '我都在想', '我认为你会有一个人的心情']:
                    continue
                if user_input in ['關機', 'exit']:
                    print("正在退出...")
                    os._exit(0)

                # 提示使用者是否繼續
                say("'{0}'('繼續'或'取消')".format(user_input))
                audio = recognizer.listen(source)
                confirmation = recognizer.recognize_whisper(audio, language="chinese", model="base")

                if '繼續' in confirmation or '继续' in confirmation:
                    answer = ask_question(llm, rag, user_input)
                    say(answer)
                else:
                    say("已取消")

            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                pass


if __name__ == '__main__':
    from llm_setup import llm_setup, rag_setup

    # 初始化 LLM
    llm = llm_setup("MediaTek-Research/Breeze-7B-Instruct-64k-v0_1")
    rag = rag_setup()

    interact(llm, rag, )
