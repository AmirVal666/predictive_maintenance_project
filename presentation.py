import streamlit as st
import reveal_slides as rs

def presentation_page():
    st.title("Презентация проекта")
    md = """
    # Прогнозирование отказов оборудования
    ---
    ## Введение
    - Цель: предсказать отказ оборудования.
    ---
    ## Этапы
    1. Загрузка и предобработка данных
    2. Обучение модели
    3. Оценка модели
    4. Streamlit-приложение
    ---
    ## Заключение
    - Точность модели, идеи для улучшений
    """
    with st.sidebar:
        theme = st.selectbox("Тема", ["black", "white", "night"])
        transition = st.selectbox("Переход", ["slide", "zoom"])
    rs.slides(md, theme=theme, config={"transition": transition})
