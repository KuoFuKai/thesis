import os
import speech_recognition as sr
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
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
                        input_variables=["context", "question"],)


def ask_question(llm, rag, question):
    say("處理中請稍後")
    rag_chain = (
            {"context": rag, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    )
    formatted_question = "{object}，{question}".format(object=variable.detected_obj, question=question)
    response = rag_chain.invoke(formatted_question)
    return response

# from langchain.chains import LLMChain
# # 定義 prompt 模板
# template = """<s>[INST] 你是導遊，遊客都只聽得懂中文，關於眼前看到的物體，若請你介紹，請你簡短介紹，若問您問題，也請您簡短回答
# 。 物體：{object}，問題：{question} [/INST] </s> """
# prompt = PromptTemplate(template=template, input_variables=["question", "object"])
#
#
# def ask_question(llm, rag, question):
#     say("處理中請稍後")
#     llm_chain = LLMChain(prompt=prompt, llm=llm)
#     response = llm_chain.invoke({"question": question, "object": variable.detected_obj})
#     return response["text"]

def interact(llm, rag):
    # 初始化語音辨識引擎
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        while True:
            # print("請問您的問題（或輸入 '關機' 退出）: ")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

            try:
                user_input = recognizer.recognize_google(audio, language='zh-TW')
                if user_input in '關機' or user_input in 'exit':
                    os._exit(0)

                say(f"您問的問題是: '{user_input}', 是否繼續？")

                confirmation = ""
                with sr.Microphone() as source2:
                    recognizer.adjust_for_ambient_noise(source2)
                    audio = recognizer.listen(source2)
                    try:
                        confirmation = recognizer.recognize_google(audio, language='zh-TW')
                        print(confirmation)
                    except sr.UnknownValueError:
                        pass  # print("Google Speech Recognition 無法理解音訊")
                    except sr.RequestError:
                        pass  # print("無法從 Google Speech Recognition 服務請求資料")

                if '繼續' in confirmation:
                    answer = ask_question(llm, rag, user_input)
                    say(answer)
                else:
                    say("已取消")

            except sr.UnknownValueError:
                pass  # print("Google Speech Recognition 無法理解音訊")
            except sr.RequestError:
                pass  # print("無法從 Google Speech Recognition 服務請求資料")
