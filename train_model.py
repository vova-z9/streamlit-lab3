import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

# 1. Генеруємо синтетичні дані для навчання
np.random.seed(42)
n_samples = 500

districts = ["Центр", "Каскад", "Пасічна", "БАМ", "Позитрон"]
conditions = ["Первинний ринок", "Євроремонт", "Житловий стан"]

data = pd.DataFrame({
    "District": np.random.choice(districts, n_samples),
    "Condition": np.random.choice(conditions, n_samples),
    "Rooms": np.random.randint(1, 5, n_samples),
    "Area": np.round(np.random.uniform(25.0, 120.0, n_samples), 1),
    "Floor": np.random.randint(1, 16, n_samples)
})

# Формула ціни оренди (в грн) для реалістичності
base_price = 6000 + data["Area"] * 120 + data["Rooms"] * 1000
district_bonus = data["District"].map({"Центр": 4000, "Каскад": 2000, "БАМ": 1500, "Пасічна": 1000, "Позитрон": 1000})
cond_bonus = data["Condition"].map({"Євроремонт": 3500, "Первинний ринок": 2500, "Житловий стан": 0})
y = base_price + district_bonus + cond_bonus + np.random.normal(0, 500, n_samples)

# 2. Створюємо і навчаємо препроцесор (масштабування чисел + кодування категорій)
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(sparse_output=False), ["District", "Condition"]),
        ("num", StandardScaler(), ["Rooms", "Area", "Floor"])
    ]
)

X_processed = preprocessor.fit_transform(data)

# 3. Створюємо просту нейронну мережу
model = Sequential([
    Input(shape=(X_processed.shape[1],)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X_processed, y, epochs=150, batch_size=16, verbose=0)

# 4. Зберігаємо модель і препроцесор
model.save("flat_price_model.keras")
joblib.dump(preprocessor, "preprocessor.pkl")
print("✅ Файли flat_price_model.keras та preprocessor.pkl успішно створені!")