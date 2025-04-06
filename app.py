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

# minor update to trigger redeploy

import streamlit as st

st.set_page_config(page_title="NutriBot", page_icon="🥦")
st.title("🥦 NutriBot: Здоровое питание")
st.write("Привет! Это тестовая версия приложения.")
