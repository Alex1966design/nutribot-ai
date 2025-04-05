import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

def load_vectorstore():
    persist_path = "vector_db"
    embedding = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    return FAISS.load_local(persist_path, embedding, allow_dangerous_deserialization=True)

def get_answer(question):
    retriever = load_vectorstore().as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(api_key=st.secrets["OPENAI_API_KEY"]),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )
    return qa_chain.run(question)
