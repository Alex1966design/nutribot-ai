from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
import os


def load_vectorstore():
    """
    Загружает локальную векторную базу из папки "vector_db" с использованием эмбеддингов OpenAI.
    """
    persist_path = "vector_db"
    # Получаем API ключ OpenAI из переменных окружения или секретов Streamlit
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    if not api_key:
        raise ValueError("OPENAI_API_KEY не найден ни в переменных окружения, ни в секретах Streamlit")

    # Создаем эмбеддинги с использованием корректного параметра
    embedding = OpenAIEmbeddings(openai_api_key=api_key)

    # Загружаем векторное хранилище, разрешая опасную десериализацию (убедитесь, что данные доверенные)
    return FAISS.load_local(persist_path, embedding, allow_dangerous_deserialization=True)


def get_answer(question, retriever):
    """
    Принимает вопрос и извлекатель retriever, генерирует ответ с помощью цепочки вопрос-ответ.
    """
    # Получаем документы, релевантные заданному вопросу
    docs = retriever.get_relevant_documents(question)

    # Получаем API ключ для ChatOpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    if not api_key:
        raise ValueError("OPENAI_API_KEY не найден ни в переменных окружения, ни в секретах Streamlit")

    # Создаем цепочку вопрос-ответ с использованием модели ChatOpenAI
    chain = load_qa_chain(ChatOpenAI(temperature=0, openai_api_key=api_key), chain_type="stuff")

    # Генерируем ответ
    answer = chain.run(input_documents=docs, question=question)
    return answer
