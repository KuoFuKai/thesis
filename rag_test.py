import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

# 配置BitsAndBytes的設定，用於模型的量化以提高效率。
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 啟用4位元量化
    bnb_4bit_compute_dtype=torch.float16,  # 計算時使用的數據類型
    bnb_4bit_quant_type="nf4",  # 量化類型
    bnb_4bit_use_double_quant=True,  # 使用雙重量化
)

# 定義模型ID，用於從HuggingFace Hub加載模型。
model_id = "MediaTek-Research/Breeze-7B-Instruct-64k-v0_1"

# 加載並配置模型，這裡使用了前面定義的量化配置。
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",  # 自動選擇運行設備
    quantization_config=quantization_config,
)

# 加載模型的分詞器。
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 創建一個用於文本生成的pipeline。
text_generation_pipeline = pipeline(
    "text-generation",
    model=model_4bit,
    tokenizer=tokenizer,
    use_cache=True,
    device_map="auto",
    max_length=10000,
    do_sample=True,
    top_k=5,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)

# 創建一個HuggingFacePipeline實例，用於後續的語言生成。
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

db = FAISS.load_local("vector_db",
                      HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'))

retriever = db.as_retriever()

# 定義一個prompt模板。
template = """<s>[INST]
你是一名導遊，遊客都只聽得懂中文，眼前看到的建築物，若請你介紹，請你簡短介紹，若問您問題，也請您簡短回答。
提示：{context}
問題：{question}
[/INST] </s> """

# 使用模板和定義的問題與上下文創建一個LLMChain實例。
prompt = PromptTemplate(template=template,
                        input_variables=["context", "question"],)

rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
)

# 定義問題和上下文。
detected_obj = "臺南孔廟石泮坊"
question = """{object}，可以請您簡介一下嗎""".format(object=detected_obj)

response = rag_chain.invoke(question)
print(response)
