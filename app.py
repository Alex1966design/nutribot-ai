import streamlit as st
from utils import load_vectorstore, get_answer

# Заголовок приложения
st.set_page_config(page_title="NutriBot — ЗОЖ ассистент", page_icon="🥦")

st.title("🥦 NutriBot: ИИ-консультант по здоровому питанию")
st.markdown("Задай любой вопрос по ЗОЖ и получи полезный совет!")

# Загружаем векторную базу один раз
@st.cache_resource
def get_retriever():
    return load_vectorstore()

retriever = get_retriever()

# Ввод вопроса
question = st.text_input("Введите ваш вопрос:", placeholder="Например: Какие продукты полезны для сердца?")

# Кнопка и ответ
if st.button("Получить ответ") and question:
    with st.spinner("Генерируем ответ..."):
        answer = get_answer(question, retriever)
        st.success("✅ Ответ:")
        st.markdown(answer)
