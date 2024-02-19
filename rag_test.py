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
model_id = "mistralai/Mistral-7B-Instruct-v0.2"

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
    max_length=500,
    do_sample=True,
    top_k=5,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)

# 創建一個HuggingFacePipeline實例，用於後續的語言生成。
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough
import nest_asyncio

nest_asyncio.apply()

# Articles to index
articles = ["https://zh.wikipedia.org/zh-tw/%E8%87%BA%E5%8D%97%E5%AD%94%E5%AD%90%E5%BB%9F",
            "https://www.twtainan.net/zh-tw/Attractions/Detail/800",
            "https://nchdb.boch.gov.tw/assets/overview/monument/19831228000007",
            "https://taiwangods.moi.gov.tw/html/landscape/1_0011.aspx?i=74#c5",
            "https://www.taiwanviptravel.com/articles/taiwan-confucian-temple/",
            "https://www.shute.kh.edu.tw/~93d31710/04.htm",
            "https://web.tainan.gov.tw/tnwcdo/News_Content.aspx?n=20016&s=6635607",
            "https://sstainan.com/tainan-destinations/tainan-confucian-temple/",
            "https://www.tn-confucius.org.tw/page.asp?orcaid=475A9094-FAEA-418F-8DCB-F39026E862DE",
            "https://www.shute.kh.edu.tw/~93d31710/13.htm", "https://carolblogtw.com/tainan-confucius-temple/",
            "https://clps10160321.weebly.com/20840214883931823416303403177720171.html",
            "https://www.findlifevalue.com/archives/5631",
            "https://findlifevalue.blogspot.com/2012/04/confucius-test1.html",
            "https://findlifevalue.blogspot.com/2012/04/confucius-test2.html",
            "https://rabbitfunaround.com/blog/post/tainan-confucius-temple"]

# Scrapes the blogs above
loader = AsyncChromiumLoader(articles)
docs = loader.load()

# Converts HTML to plain text
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)

# Chunk text
text_splitter = CharacterTextSplitter(chunk_size=100,
                                      chunk_overlap=0)
chunked_documents = text_splitter.split_documents(docs_transformed)

# Load chunked documents into the FAISS index
db = FAISS.from_documents(chunked_documents,
                          HuggingFaceEmbeddings(
                              model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'))

retriever = db.as_retriever()

# 定義一個prompt模板。
template = """<s>[INST] 你是導遊，遊客都只聽得懂中文，關於眼前看到的物體，若請你介紹，請你簡短介紹，若問您問題，也請您簡短回答
。 物體：台南孔子廟，問題：{question} [/INST] </s> """

# 使用模板和定義的問題與上下文創建一個LLMChain實例。
prompt = PromptTemplate(template=template, input_variables=["question", "object"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} | llm_chain
)

# 定義問題和上下文。
question = """可以請您簡介嗎"""

response = rag_chain.invoke(question)["text"]
print(response)
