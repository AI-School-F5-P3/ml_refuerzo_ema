import streamlit as st
import pandas as pd
import requests
import os
from PIL import Image

# Configuración de la URL de la API
API_URL = "http://127.0.0.1:8000/predict/"  # Cambia si tu API usa otra dirección o puerto

# Agregar logo en la parte superior de la página
logo_path = "../src/images/logo.png"  # Ruta al logo
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.image(logo, width=100)  # Ajusta el tamaño del logo según sea necesario

# Título de la aplicación
st.title("Clasificación de Clientes")

# Menú lateral
menu = st.sidebar.selectbox(
    "Selecciona una opción",
    [
        "Introducción",
        "Mostrar Gráficas",
        "Mostrar Métricas",
        "Comparar Rendimientos",
        "Predecir Grupo de un Cliente"
    ]
)

# Función para ingresar los datos de un cliente
def user_input_features():
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Edad", min_value=1, max_value=80, step=1)
        gender = st.selectbox("Género", [0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
        marital = st.selectbox("Estado Civil", [0, 1], format_func=lambda x: "No Casado" if x == 0 else "Casado")
        ed = st.number_input("Nivel Educativo (1 a 5)", min_value=1, max_value=5, step=1)
        reside = st.number_input("Años en Residencia", min_value=1, max_value=8, step=1)

    with col2:
        region = st.number_input("Región", min_value=1, max_value=3, step=1)
        tenure = st.number_input("Tenencia", min_value=1, max_value=100, step=1)
        address = st.number_input("Dirección (años en la dirección)", min_value=0, max_value=60, step=1)
        income = st.number_input("Ingresos", format="%.2f")
        employ = st.number_input("Años de Empleo", min_value=1, max_value=5, step=1)

    with col3:
        retire = st.selectbox("¿Está Jubilado?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
        income_per_year_employed = st.number_input("Ingresos por Año Empleado", format="%.2f")
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

elif menu == "Predecir Grupo de un Cliente":
    st.markdown("## Predecir Grupo de un Cliente")
    input_df = user_input_features()

    if st.button("Predecir Grupo"):
        try:
            # Realizar la solicitud POST a la API
            response = requests.post(API_URL, json=input_df.iloc[0].to_dict())
            
            if response.status_code == 200:
                result = response.json()
                predicted_group = result["predicted_group"]
                predicted_probability = result["probability"]

                # Mostrar los resultados
                st.success(f"El cliente pertenece al grupo: {predicted_group}")
                st.write(f"Probabilidad de pertenecer al grupo {predicted_group}: {predicted_probability:.2%}")
            else:
                st.error(f"Error en la API: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Error al conectar con la API: {e}")

# Otros menús pueden mantener su lógica actual.
elif menu == "Mostrar Gráficas":
    st.markdown("## Mostrar Gráficas")
    st.markdown("Esta funcionalidad no está integrada con la API.")

elif menu == "Mostrar Métricas":
    st.markdown("## Mostrar Métricas")
    st.markdown("Esta funcionalidad no está integrada con la API.")

elif menu == "Comparar Rendimientos":
    st.markdown("## Comparar Rendimientos")
    st.markdown("Esta funcionalidad no está integrada con la API.")

