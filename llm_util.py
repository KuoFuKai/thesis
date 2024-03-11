import os
import variable
import speech_recognition as sr
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
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
    variable.pause_interact_event.clear()

    say("處理中請稍後")
    formatted_question = "{object}，{question}".format(object=variable.detected_obj, question=question)
    rag_chain = RetrievalQA.from_chain_type(llm=llm,
                                            retriever=rag,
                                            chain_type="stuff",
                                            chain_type_kwargs={"prompt": prompt}, )
    answer = rag_chain.invoke({"query": formatted_question})["result"]

    variable.pause_interact_event.set()
    return answer


question_prefix_words = ['hi', '嗨', '害', '愛', '太', '泰']
continue_prefix_word = ['yes', 'no']


def interact(llm, rag):
    # 初始化語音辨識引擎
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        while True:
            variable.pause_interact_event.wait()
            print("Ask a question（or say '關機' to exit）: ")
            audio = recognizer.listen(source)

            try:
                user_input = recognizer.recognize_whisper(audio, model="base").lower()
                # user_input = recognizer.recognize_google(audio, language='zh-TW').lower()
                if any(user_input.startswith(prefix) for prefix in question_prefix_words):
                    for prefix in question_prefix_words:
                        if user_input.startswith(prefix):
                            user_input = user_input[len(prefix):].lstrip()
                            break
                else:
                    print(user_input)
                    if user_input in ['繼續', '继续']:
                        variable.pause_detect_event.clear()

                    if user_input in ['關機', 'exit']:
                        print("正在退出...")
                        os._exit(0)

                    continue

                # 提示使用者是否繼續
                say("'{0}'(yes or no)".format(user_input))
                while True:
                    audio = recognizer.listen(source)
                    confirmation = recognizer.recognize_whisper(audio, model="base").lower().strip('. ')
                    # confirmation = recognizer.recognize_google(audio, language='zh-TW').lower().strip('. ')
                    print(confirmation)
                    if 'yes' in confirmation:
                        answer = ask_question(llm, rag, user_input)
                        say(answer)
                        break
                    elif 'no' in confirmation:
                        say("已取消")
                        break

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
