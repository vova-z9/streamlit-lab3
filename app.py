import joblib
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

# 1. Завантаження моделі та препроцесора з кешуванням
@st.cache_resource
def load_model_and_resources():
    model = load_model("flat_price_model.keras", compile=False)
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_model_and_resources()

# 2. Інтерфейс користувача
st.title("🏠 Прогнозування вартості оренди квартири")
st.write("Введіть параметри житла, щоб нейронна мережа розрахувала орієнтовну місячну орендну плату.")

col1, col2 = st.columns(2)

with col1:
    district = st.selectbox("Район", ["Центр", "Каскад", "Пасічна", "БАМ", "Позитрон"])
    condition = st.radio("Стан ремонту", ["Первинний ринок", "Євроремонт", "Житловий стан"])

with col2:
    rooms = st.number_input("Кількість кімнат", min_value=1, max_value=5, value=2, step=1)
    area = st.number_input("Площа (м²)", min_value=20.0, max_value=150.0, value=50.0, step=1.0)

floor = st.slider("Поверх", min_value=1, max_value=16, value=4, step=1)

# 3. Попередня обробка даних
def preprocess_input(district, condition, rooms, area, floor, preprocessor):
    df = pd.DataFrame({
        "District": [district],
        "Condition": [condition],
        "Rooms": [rooms],
        "Area": [area],
        "Floor": [floor]
    })
    return preprocessor.transform(df)

# 4. Прогнозування
if st.button("Розрахувати вартість"):
    input_data = preprocess_input(district, condition, rooms, area, floor, preprocessor)
    pred = model.predict(input_data, verbose=0)
    st.success(f"Орієнтовна вартість оренди: {pred[0][0]:,.0f} грн / міс.")