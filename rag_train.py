from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import nest_asyncio

nest_asyncio.apply()

# Articles to index
articles = ["https://www.tn-confucius.org.tw/page.asp?orcaid=E085B352-A7C6-4838-86C4-D1DFB9771685",
            "https://taiwangods.moi.gov.tw/html/landscape/1_0011.aspx?i=74",
            "https://nrch.culture.tw/twpedia.aspx?id=2672",
            "https://zh.wikipedia.org/zh-tw/%E8%87%BA%E5%8D%97%E5%AD%94%E5%AD%90%E5%BB%9F",
            "https://www.tn-confucius.org.tw/page.asp?orcaid=475A9094-FAEA-418F-8DCB-F39026E862DE",
            "https://zh.wikipedia.org/zh-tw/%E8%87%BA%E5%8D%97%E5%AD%94%E5%AD%90%E5%BB%9F%E6%98%8E%E5%80%AB%E5%A0%82",
            "https://confucius.culture.tw/home/zh-tw/map/34409",
            "https://memory.culture.tw/Home/Detail?Id=600598&IndexCode=online_metadata",
            "https://zh.wikipedia.org/zh-tw/%E6%98%8E%E4%BC%A6%E5%A0%82",
            "https://tprn.news/2018/01/22/1540/",
            "https://tcmb.culture.tw/zh-tw/detail?indexCode=Culture_Object&id=112657",
            "https://memory.culture.tw/Home/Detail?Id=112657&IndexCode=Culture_Object",
            "https://tprn.news/2018/02/02/2400/",
            "http://www.gloje-chea.org/extend_reading.php?rely_id=43",
            "https://tprn.news/2018/01/08/1638/",
            "https://blog.udn.com/tnwinsto/132400793",
            "http://deh.csie.ncku.edu.tw/extn/poi_detail/32362",
            "https://madge1022.pixnet.net/blog/post/41237051",
            "https://www.facebook.com/tnfzst/posts/%E6%9C%8B%E5%8F%8B%E5%80%91%E4%BE%86%E5%AD%94%E5%BB%9F%E5%95%86%E5%9C%88%E4%B8%80%E5%AE%9A%E6%9C%83%E7%9C%8B%E5%88%B0%E9%80%99%E5%BA%A7%E6%B3%AE%E5%AE%AE%E7%9F%B3%E5%9D%8A%E6%82%A8%E7%9F%A5%E9%81%93%E4%BB%96%E7%9A%84%E4%BE%86%E6%AD%B7%E8%83%8C%E6%99%AF%E5%97%8E%E4%B8%80%E8%B5%B7%E4%BE%86%E4%BA%86%E8%A7%A3%E6%B3%AE%E5%AE%AB%E7%9A%84%E6%95%85%E4%BA%8B%E5%90%A7/1735176736768766/",
            "https://asc.tainan.gov.tw/index.php?modify=place&id=6"]

# Scrapes the blogs above
loader = AsyncChromiumLoader(articles)
docs = loader.load()

# Converts HTML to plain text
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)

# Chunk text
text_splitter = CharacterTextSplitter(chunk_size=1024,
                                      chunk_overlap=0)
chunked_documents = text_splitter.split_documents(docs_transformed)

# Load chunked documents into the FAISS index
db = FAISS.from_documents(chunked_documents,
                          HuggingFaceEmbeddings(
                              model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'))

db.save_local("vector_db")
