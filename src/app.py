import streamlit as st
import pandas as pd
import pickle

# Configuración de la página
st.set_page_config(page_title="Clasificación de Clientes", layout="centered")

# Título de la aplicación
st.title("Clasificación de Clientes")

# Descripción
st.markdown("**Ingresa las características del cliente para determinar el grupo al que pertenece.**")

# Entrada de datos del usuario
def user_input_features():
    region = st.number_input("Region", min_value=1, max_value=3, step=1)
    tenure = st.number_input("Tenure", min_value=1, max_value=100, step=1)
    age = st.number_input("Age", min_value=1, max_value=80, step=1)
    marital = st.selectbox("Marital Status", [0, 1], format_func=lambda x: "No Casado" if x == 0 else "Casado")
    address = st.number_input("Address (años en la dirección)", min_value=0, max_value=60, step=1)
    income = st.number_input("Income", format="%.1f")
    ed = st.number_input("Nivel Educativo (1 a 5)", min_value=1, max_value=5, step=1)
    employ = st.number_input("Años de Empleo", min_value=1, max_value=5, step=1)
    retire = st.selectbox("¿Está Jubilado?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
    gender = st.selectbox("Género", [0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
    reside = st.number_input("Años en Residencia", min_value=1, max_value=8, step=1)
    income_per_year_employed = st.number_input("Income por Año Empleado", format="%.1f")
    age_category = st.number_input("Categoría de Edad", min_value=1, max_value=4, step=1)
    income_age_ratio = st.number_input("Relación Ingreso/Edad", format="%.1f")

    # Creación del DataFrame con las características ingresadas
    data = {
        "region": region,
        "tenure": tenure,
        "age": age,
        "marital": marital,
        "address": address,
        "income": income,
        "ed": ed,
        "employ": employ,
        "retire": retire,
        "gender": gender,
        "reside": reside,
        "income_per_year_employed": income_per_year_employed,
        "age_category": age_category,
        "income_age_ratio": income_age_ratio
    }
    return pd.DataFrame(data, index=[0])

# Obtener las características
input_df = user_input_features()

# Cargar el modelo KNN desde el archivo knn.pkl
@st.cache_resource
def load_model():
    try:
        with open("../src/models/logistic_regression.pkl", "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Archivo 'logistic_regression.pkl' no encontrado. Asegúrate de que está en el directorio correcto.")
        return None

model = load_model()

# Predicción
if st.button("Predecir Grupo"):
    if model is not None:
        try:
            prediction = model.predict(input_df)  # Realiza la predicción
            probabilities = model.predict_proba(input_df)  # Obtiene las probabilidades

            # Obtener la probabilidad del grupo predicho
            predicted_group = int(prediction[0])
            predicted_probability = probabilities[0][predicted_group]

            # Mostrar el resultado
            st.success(f"El cliente pertenece al grupo: {predicted_group}")
            st.write(f"Probabilidad de pertenecer al grupo {predicted_group}: {predicted_probability:.2%}")
        except Exception as e:
            st.error(f"Error al realizar la predicción: {e}")
    else:
        st.error("El modelo no se pudo cargar correctamente. Revisa el archivo 'logistic_regression.pkl'.")