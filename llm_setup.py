import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def llm_setup(model_id):
    # 配置BitsAndBytes的設定
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # 加載模型和分詞器
    model_4bit = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 創建pipeline
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model_4bit,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=1500,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline, )

    return llm

def rag_setup():
    db = FAISS.load_local("vector_db",
                          HuggingFaceEmbeddings(
                              model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'))
    return db.as_retriever()
