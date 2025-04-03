# utils.py
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

# Путь к файлу и базе
DATA_PATH = "data/zdorovoe_info.md"
DB_DIR = "vector_db"

# Загрузка и создание векторной базы
def load_vectorstore():
    if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
        print("Создаю новую векторную базу...")
        loader = TextLoader(DATA_PATH, autodetect_encoding=True)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". "]
        )
        docs = splitter.split_documents(documents)

        embedding = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(docs, embedding=embedding, persist_directory=DB_DIR)
        vectordb.persist()
    else:
        print("Загружаю существующую векторную базу...")
        embedding = OpenAIEmbeddings()
        vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embedding)

    return vectordb.as_retriever()

# Получение ответа на вопрос
def get_answer(question, retriever):
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o", temperature=0.5),
        retriever=retriever,
        return_source_documents=False
    )
    return qa_chain.run(question)