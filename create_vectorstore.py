from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

# Загружаем и разбиваем текст
loader = TextLoader("data/zdorovoe_info.md", encoding="utf-8")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Генерация эмбеддингов
embedding = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])

# Создание векторной базы и сохранение
vectorstore = FAISS.from_documents(texts, embedding)
vectorstore.save_local("vector_db")
