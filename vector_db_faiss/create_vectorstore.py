from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import os

# Загрузка и разбиение исходного текста
loader = TextLoader("data/zdorovoe_info.md", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)

# Генерация эмбеддингов и создание FAISS-векторной базы
embedding = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = FAISS.from_documents(docs, embedding)

# Сохраняем векторную базу
vectorstore.save_local("vector_db_faiss")

print("✅ FAISS-векторная база успешно создана и сохранена.")
