import os
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import streamlit as st
from tempfile import TemporaryDirectory

CHROMA_SETTINGS = {
    "collection_name": "nutri_collection",
}

# Временная директория для Streamlit Cloud
persist_directory = TemporaryDirectory().name

def create_vectorstore(file_path: str):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embedding = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])

    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=persist_directory,
        client_settings=CHROMA_SETTINGS,
    )
    vectorstore.persist()
    return vectorstore

def load_vectorstore():
    embedding = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return retriever

def get_answer(retriever, question):
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(api_key=st.secrets["OPENAI_API_KEY"]),
        retriever=retriever,
        return_source_documents=False
    )
    return qa_chain.run(question)
