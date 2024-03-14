import torch
from auto_gptq import BaseQuantizeConfig, AutoGPTQForCausalLM
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def llm_setup(model_id):
    # # 配置BitsAndBytes的設定
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    # )
    #
    # # 加載模型和分詞器
    # model_4bit = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     device_map="auto",
    #     quantization_config=quantization_config,
    # )
    # from vllm import LLM, SamplingParams
    # sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=256)
    # llm = LLM(
    #     model=model_id,
    #     quantization='awq',
    #     dtype='half',
    # )

    # model_4bit = AutoModelForCausalLM.from_pretrained(
    #         model_id,
    #         device_map="auto",
    #         use_safetensors=True,
    #     )

    quantize_config = BaseQuantizeConfig(
    )

    model_4bit = AutoGPTQForCausalLM.from_quantized(
        model_id,
        device="cuda:0",)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    # 創建pipeline
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
        streamer=streamer,
    )

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline,
                              callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                              )

    return llm


def rag_setup():
    db = FAISS.load_local("vector_db",
                          HuggingFaceEmbeddings(
                              model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'))
    return db.as_retriever()
