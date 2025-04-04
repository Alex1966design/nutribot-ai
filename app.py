import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI


def load_vectorstore():
    print("Загружаю векторную базу...")

    # Загрузка Markdown-файлов из папки data/
    text_loader_kwargs = {'autodetect_encoding': True}
    loader = DirectoryLoader("data", glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    documents = loader.load()

    # Разбиение текста на чанки
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "(?<=\. )"]
    )
    splitted_texts = splitter.split_documents(documents)

    # Создание эмбеддингов и векторной базы в памяти
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=splitted_texts,
        embedding=embedding
        # persist_directory не указываем — работаем в in-memory
    )

    # Возвращаем retriever для поиска по базе
    return vectordb.as_retriever()


def get_answer(retriever, question):
    # Создание модели и цепочки RetrievalQA
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Генерация ответа на вопрос
    return qa_chain.run(question)
# версия от 2024-04-03
