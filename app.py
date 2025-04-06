import streamlit as st

# Кнопка для сброса кэша
if st.button("Очистить кэш"):
    st.cache_resource.clear()
    st.experimental_memo.clear()
    st.success("Кэш очищен!")

import streamlit as st
from utils import load_vectorstore, get_answer

# Загружаем векторную базу один раз и кэшируем результат
@st.cache_resource
def get_retriever():
    return load_vectorstore()

# Получаем извлекатель (не передаём его в get_answer, т.к. функция принимает только вопрос)
retriever = get_retriever()

# Пользовательский ввод вопроса
question = st.text_input("Введите ваш вопрос:", placeholder="Например: Какие продукты полезны для сердца?")

# Если нажата кнопка и введён вопрос, генерируем ответ
if st.button("Получить ответ") and question:
    with st.spinner("Генерируем ответ..."):
        answer = get_answer(question)  # Передаём только вопрос
    st.success("✅ Ответ:")
    st.markdown(answer)
