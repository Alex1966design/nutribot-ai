import os
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_core.documents import Document
import streamlit as st


@st.cache_resource
def load_vectorstore():
    embedding = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    persist_path = "vector_db_faiss"
    if os.path.exists(persist_path):
        return FAISS.load_local(persist_path, embedding)
    else:
        raise ValueError(f"Векторная база данных FAISS не найдена по пути: {persist_path}")


def get_answer(question: str):
    embedding = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=st.secrets["OPENAI_API_KEY"])
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain.run(question)
