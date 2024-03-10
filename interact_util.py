import variable
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
    # variable.pause_interact_event.set()

    say("處理中請稍後")
    formatted_question = "{object}，{question}".format(object=variable.detected_obj, question=question)
    rag_chain = RetrievalQA.from_chain_type(llm=llm,
                                            retriever=rag,
                                            chain_type="stuff",
                                            chain_type_kwargs={"prompt": prompt}, )
    answer = rag_chain.invoke({"query": formatted_question})["result"]

    # variable.pause_interact_event.clear()
    return answer

