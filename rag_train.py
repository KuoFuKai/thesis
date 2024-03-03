from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
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
text_splitter = CharacterTextSplitter(chunk_size=1024,
                                      chunk_overlap=0)
chunked_documents = text_splitter.split_documents(docs_transformed)

# Load chunked documents into the FAISS index
db = FAISS.from_documents(chunked_documents,
                          HuggingFaceEmbeddings(
                              model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'))

db.save_local("vector_db")
