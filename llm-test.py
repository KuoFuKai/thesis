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

# 定義一個prompt模板。
template = """<s>[INST] You are a helpful, respectful and honest assistant. and I want to ask you a simple question :
{question} [/INST] </s>
"""

# 定義問題和上下文。
question_p = """can you explain tiger to me"""

# 使用模板和定義的問題與上下文創建一個LLMChain實例。
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

# 執行LLMChain，得到回應。
response = llm_chain.run({"question": question_p})
print(response)

# # 定義一個prompt模板。
# template = """<s>[INST] You are a helpful, respectful and honest assistant. Answer exactly in few words from the context
# Answer the question below from context below :
# {context}
# {question} [/INST] </s>
# """
#
# # 定義問題和上下文。
# question_p = """What is the date for announcement"""
# context_p = """ On August 15 said that its arm JSW Neo Energy has agreed to buy a portfolio of 1753 mega watt renewable energy generation capacity from Mytrah Energy India Pvt Ltd for Rs 10,530 crore."""
#
# # 使用模板和定義的問題與上下文創建一個LLMChain實例。
# prompt = PromptTemplate(template=template, input_variables=["question", "context"])
# llm_chain = LLMChain(prompt=prompt, llm=llm)
#
# # 執行LLMChain，得到回應。
# response = llm_chain.run({"question": question_p, "context": context_p})
# print(response)
