import streamlit as st
import pandas as pd
import pickle

# Configuración de la página
st.set_page_config(page_title="Clasificación de Clientes", layout="wide")

# Título de la aplicación
st.title("Clasificación de Clientes")

# Menú lateral
menu = st.sidebar.selectbox(
    "Selecciona una opción",
    [
        "Introducción",
        "Mostrar Métricas",
        "Elegir Modelo",
        "Comparar Rendimientos",
        "Predecir Grupo de un Cliente"
    ]
)

# Función para cargar un modelo
@st.cache_resource
def load_model(model_path):
    try:
        with open(model_path, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"Archivo '{model_path}' no encontrado.")
        return None

# Función para ingresar los datos de un cliente
def user_input_features():
    col1, col2 = st.columns(2)

    with col1:
        region = st.number_input("Region", min_value=1, max_value=3, step=1)
        tenure = st.number_input("Tenure", min_value=1, max_value=100, step=1)
        age = st.number_input("Age", min_value=1, max_value=80, step=1)
        marital = st.selectbox("Marital Status", [0, 1], format_func=lambda x: "No Casado" if x == 0 else "Casado")
        address = st.number_input("Address (años en la dirección)", min_value=0, max_value=60, step=1)
        income = st.number_input("Income", format="%.2f")
        ed = st.number_input("Nivel Educativo (1 a 5)", min_value=1, max_value=5, step=1)

    with col2:
        employ = st.number_input("Años de Empleo", min_value=1, max_value=5, step=1)
        retire = st.selectbox("¿Está Jubilado?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
        gender = st.selectbox("Género", [0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
        reside = st.number_input("Años en Residencia", min_value=1, max_value=8, step=1)
        income_per_year_employed = st.number_input("Income por Año Empleado", format="%.2f")
        age_category = st.number_input("Categoría de Edad", min_value=1, max_value=4, step=1)
        income_age_ratio = st.number_input("Relación Ingreso/Edad", format="%.2f")

    # Crear DataFrame con los datos ingresados
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

# Lógica de las opciones del menú
if menu == "Introducción":
    st.markdown("## Introducción")
    st.markdown(
        "Esta aplicación permite clasificar a clientes en diferentes grupos "
        "según sus características, evaluar métricas de rendimiento de los modelos, "
        "y comparar el desempeño de diferentes algoritmos."
    )

elif menu == "Mostrar Métricas":
    st.markdown("## Métricas del Modelo")
    st.markdown("Aquí puedes mostrar métricas como precisión, recall, F1, etc.")
    # Este bloque debería incluir la lógica para cargar y mostrar las métricas.

elif menu == "Elegir Modelo":
    st.markdown("## Selección de Modelo")
    st.markdown("Selecciona un modelo para realizar las predicciones:")
    model_option = st.selectbox(
        "Modelos Disponibles",
        ["Logistic Regression", "K-Nearest Neighbors", "Random Forest"]
    )

    if model_option == "Logistic Regression":
        model_path = "../src/models/logistic_regression.pkl"
    elif model_option == "K-Nearest Neighbors":
        model_path = "../src/models/knn.pkl"
    elif model_option == "Random Forest":
        model_path = "../src/models/random_forest.pkl"
    else:
        model_path = None

    if model_path:
        model = load_model(model_path)
        if model is not None:
            st.success(f"Modelo '{model_option}' cargado con éxito.")
        else:
            st.error("Error al cargar el modelo. Verifica la ruta.")

elif menu == "Comparar Rendimientos":
    st.markdown("## Comparar Rendimientos")
    st.markdown("En este apartado puedes comparar el rendimiento de diferentes modelos.")
    # Este bloque debería incluir lógica para graficar comparaciones de métricas.

elif menu == "Predecir Grupo de un Cliente":
    st.markdown("## Predecir Grupo de un Cliente")
    input_df = user_input_features()

    # Modelo por defecto (puedes personalizarlo o conectarlo con la opción 'Elegir Modelo')
    default_model_path = "../src/models/logistic_regression.pkl"
    model = load_model(default_model_path)

    if model is not None:
        if st.button("Predecir Grupo"):
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
        st.error("No se pudo cargar el modelo. Verifica el archivo del modelo.")
