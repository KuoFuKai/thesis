import multiprocessing
import os
import threading
import gc
import torch
from multiprocessing import Queue

import speech_recognition as sr
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import variable
from tts_util import say, say_queue

# 定義一個prompt模板。
template = """<s>[INST]
你是一名導遊，遊客都只聽得懂中文，眼前看到的建築物，請您回覆精確、簡短，不要提到問題外的事物。
提示：{context}
問題：{question}
[/INST] </s> """
# 使用模板和定義的問題與上下文創建一個LLMChain實例。
prompt = PromptTemplate(template=template,
                        input_variables=["context", "question"], )

ask_process = None

def ask_question(llm, rag, streamer, question):
    variable.pause_interact_event.clear()
    global ask_process
    if ask_process and ask_process.is_alive():
        ask_process.terminate()
        ask_process.join()
        ask_process = None

    # say("處理中請稍後")
    formatted_question = "{object}，{question}".format(object=variable.detected_obj, question=question)
    rag_chain = RetrievalQA.from_chain_type(llm=llm,
                                            retriever=rag,
                                            chain_type="stuff",
                                            chain_type_kwargs={"prompt": prompt}, )
    thread = threading.Thread(target=rag_chain.invoke, kwargs={"input": {"query": formatted_question}})
    thread.start()

    interact_queue = Queue()
    ask_process = multiprocessing.Process(target=say_queue, args=(interact_queue,))
    ask_process.start()
    current_sentence = ""
    for token in streamer:
        print(token, end='', flush=True)
        current_sentence += token
        if token in ['\n', '，', '。', '！', ',', '.']:  # 根据需要添加其他终止符
            interact_queue.put(current_sentence)
            current_sentence = ""
    interact_queue.put(None)
    variable.pause_interact_event.set()


question_prefix_words = ['Hi', 'hi', '嗨', '害', '愛', '太', '泰']
continue_prefix_word = ['yes', 'no']


def interact(llm, rag, streamer):
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
                            if any(substring in user_input for substring in ['繼續', '继续']):
                                variable.pause_detect_event.clear()
                                continue
                            break
                else:
                    print(user_input)
                    if any(substring in user_input for substring in ['繼續', '继续']):
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
                        answer = ask_question(llm, rag, streamer, user_input)
                        say(answer)
                        torch.cuda.empty_cache()
                        gc.collect()
                        break
                    elif 'no' in confirmation:
                        say("已取消")
                        break

            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                pass


if __name__ == '__main__':
    from llm_setup import tokenizer_setup, streamer_setup, llm_setup, rag_setup

    # 初始化 LLM
    llm_model = "MediaTek-Research/Breeze-7B-Instruct-64k-v0_1"
    tokenizer = tokenizer_setup(llm_model)
    streamer = streamer_setup(tokenizer)
    llm = llm_setup(llm_model, tokenizer, streamer)
    rag = rag_setup()

    interact(llm, rag, streamer, )
