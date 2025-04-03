# 🥦 NutriBot AI — ИИ-ассистент по здоровому питанию

NutriBot — это интерактивное Streamlit-приложение, в котором используется LangChain, OpenAI и ChromaDB для ответа на вопросы по теме здорового питания.

---

## 🚀 Возможности
- 🤖 Отвечает на вопросы, используя векторную базу знаний
- 📚 Работает с пользовательскими текстами и статьями
- 🧠 Использует GPT-4o для генерации точных и релевантных ответов
- 📦 Готов к развёртыванию на Streamlit Cloud или GitHub Pages

---

## 🛠 Стек технологий
- [Streamlit](https://streamlit.io)
- [LangChain](https://www.langchain.com/)
- [OpenAI API (GPT-4o)](https://platform.openai.com/)
- [ChromaDB](https://www.trychroma.com/)

---

## 📂 Структура проекта
```
nutribot/
├── app.py                  # интерфейс Streamlit
├── utils.py                # логика поиска и генерации
├── requirements.txt        # зависимости
├── README.md               # описание проекта
├── data/
│   └── zdorovoe_info.md    # материалы по ЗОЖ
└── vector_db/              # векторная база (создаётся автоматически)
```

---

## 🧪 Как запустить локально

```bash
pip install -r requirements.txt
streamlit run app.py
```

Приложение откроется по адресу: `http://localhost:8501`

---

## 🌐 Демо-версия

👉 [Streamlit Cloud](https://streamlit.io/cloud) (можно развернуть за 1 минуту)

---

## 📬 Контакты
Проект создан в рамках изучения LangChain + ИИ. Если хотите интегрировать ИИ в свой бизнес — пишите!

📧 Email: info@zosh-expert.ru

---

Сделано с любовью к ЗОЖ ❤️ и ИИ 🤖
