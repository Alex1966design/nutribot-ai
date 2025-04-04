from utils import load_vectorstore, get_answer

# Загружаем векторную базу
retriever = load_vectorstore()

# Пробный вопрос
question = "Какие продукты полезны для сердца?"

# Получаем ответ
answer = get_answer(question, retriever)

# Выводим ответ
print("🧠 Ответ:", answer)
