from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI


def load_vectorstore():
    """
    Загружает локальную векторную базу из папки "vector_db" с использованием эмбеддингов OpenAI.
    """
    persist_path = "vector_db"
    # Получаем эмбеддинги с ключом из секретов Streamlit
    embedding = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    # Загружаем векторное хранилище, разрешая опасную десериализацию (убедитесь, что вы доверяете данным)
    return FAISS.load_local(persist_path, embedding, allow_dangerous_deserialization=True)


def get_answer(question):
    """
    Принимает вопрос и генерирует ответ, используя цепочку вопрос-ответ.
    """
    # Получаем извлекатель (retriever) из векторного хранилища
    retriever = load_vectorstore().as_retriever(search_kwargs={"k": 3})

    # Создаем цепочку вопрос-ответ с использованием модели ChatOpenAI
    chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff")

    # Получаем документы, релевантные вопросу
    docs = retriever.get_relevant_documents(question)

    # Генерируем ответ с помощью цепочки
    answer = chain.run(input_documents=docs, question=question)
    return answer
