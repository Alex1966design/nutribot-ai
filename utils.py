def get_answer(question, retriever):
    # Здесь используется retriever для поиска информации по вопросу
    result = retriever.get_relevant_documents(question)
    answer = result[0]["text"] if result else "Ответ не найден."
    return answer
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

    from langchain.chains.question_answering import load_qa_chain
    from langchain_community.chat_models import ChatOpenAI

    def get_answer(question, retriever):
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
        chain = load_qa_chain(llm, chain_type="stuff")
        docs = retriever.get_relevant_documents(question)
        return chain.run(input_documents=docs, question=question)

    )
    return qa_chain.run(question)
