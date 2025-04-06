import streamlit as st
from utils import get_answer, get_retriever

# Заголовок
st.title("NutriBot: ИИ-консультант по здоровому питанию")
st.write("Задай любой вопрос по ЗОЖ и получи полезный совет!")

# Ввод вопроса
question = st.text_input("Введите ваш вопрос:", placeholder="Например: Какие продукты полезны для сердца?")

# Обработчик нажатия кнопки
if st.button("Получить ответ") and question:
    with st.spinner("Генерируем ответ..."):
        retriever = get_retriever()  # получаем retriever (загружаем векторную базу)
        answer = get_answer(question, retriever)  # передаем оба параметра
        st.success("✅ Ответ:")
        st.markdown(answer)
import streamlit as st
from utils import get_answer

# Заголовок приложения
st.title("NutriBot: ИИ-консультант по здоровому питанию")
st.subheader("Задай любой вопрос по ЗОЖ и получи полезный совет!")

# Ввод вопроса
question = st.text_input("Введите ваш вопрос:", placeholder="Например: Какие продукты полезны для сердца?")

# Кнопка и ответ
if st.button("Получить ответ") and question:
    with st.spinner("Генерируем ответ..."):
        # Генерация ответа
        answer = get_answer(question)  # Теперь передается только один параметр
    st.success("✅ Ответ:")
    st.markdown(answer)

