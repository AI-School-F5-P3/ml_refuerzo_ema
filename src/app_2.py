import streamlit as st
import pandas as pd
import pickle
import os
from PIL import Image

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

# Función para cargar un modelo
@st.cache_resource
def load_model(model_path):
    try:
        with open(model_path, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"Archivo '{model_path}' no encontrado.")
        return None
    
# Función para cargar las métricas
def load_metrics(metrics_dir):
    metrics_files = [f for f in os.listdir(metrics_dir) if f.endswith('.csv')]
    metrics = {}
    for file in metrics_files:
        model_name = os.path.splitext(file)[0]  # El nombre del modelo
        metrics[model_name] = pd.read_csv(os.path.join(metrics_dir, file))
    return metrics

# Función para ingresar los datos de un cliente (en tres columnas)
def user_input_features():
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=80, step=1)
        gender = st.selectbox("Género", [0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
        marital = st.selectbox("Marital Status", [0, 1], format_func=lambda x: "No Casado" if x == 0 else "Casado")
        ed = st.number_input("Nivel Educativo (1 a 5)", min_value=1, max_value=5, step=1)
        reside = st.number_input("Años en Residencia", min_value=1, max_value=8, step=1)

    with col2:
        region = st.number_input("Region", min_value=1, max_value=3, step=1)
        tenure = st.number_input("Tenure", min_value=1, max_value=100, step=1)
        address = st.number_input("Address (años en la dirección)", min_value=0, max_value=60, step=1)
        income = st.number_input("Income", format="%.2f")
        employ = st.number_input("Años de Empleo", min_value=1, max_value=5, step=1)

    with col3:
        retire = st.selectbox("¿Está Jubilado?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
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
elif menu == "Mostrar Gráficas":
    st.markdown("## Gráficas de los Modelos")
    
    # Ruta al directorio donde se almacenan las gráficas
    metrics_dir = "../src/metrics"  # Cambiar a la ruta correcta
    model_graphs = {}

    # Buscar archivos de gráficos por modelo
    for file in os.listdir(metrics_dir):
        if file.endswith(".png"):
            model_name = file.split("_")[0]  # Extraer el nombre del modelo (antes del guion bajo)
            if model_name not in model_graphs:
                model_graphs[model_name] = []
            model_graphs[model_name].append(os.path.join(metrics_dir, file))
    
    # Mostrar gráficas por modelo en pestañas
    if model_graphs:
        tabs = st.tabs(list(model_graphs.keys()))  # Crear pestañas para cada modelo
        for i, model_name in enumerate(model_graphs.keys()):
            with tabs[i]:
                st.markdown(f"### Gráficas para el modelo: {model_name}")
                for graph_path in model_graphs[model_name]:
                    st.image(graph_path, caption=os.path.basename(graph_path))
    else:
        st.error("No se encontraron archivos de gráficas en el directorio '../src/metrics'.")

        
elif menu == "Mostrar Métricas":
    st.markdown("## Métricas de los Modelos")
    metrics_dir = "../src/metrics"
    metrics = load_metrics(metrics_dir)

    if metrics:
        tabs = st.tabs(list(metrics.keys()))  # Crear pestañas para cada modelo
        for i, model_name in enumerate(metrics.keys()):
            with tabs[i]:
                st.markdown(f"### Métricas para el modelo: {model_name}")
                st.dataframe(metrics[model_name])
    else:
        st.error("No se encontraron métricas en el directorio '../src/metrics'.")

elif menu == "Comparar Rendimientos":
    st.markdown("## Comparar Rendimientos de Modelos")
    comparison_file = "../src/metrics/models_comparision.csv"
    try:
        comparison_df = pd.read_csv(comparison_file)
        st.dataframe(comparison_df)

        # Graficar comparaciones
        st.markdown("### Gráficas Comparativas")
        st.line_chart(comparison_df.set_index("Modelo"))
    except FileNotFoundError:
        st.error(f"No se encontró el archivo '{comparison_file}'.")

elif menu == "Predecir Grupo de un Cliente":
    st.markdown("## Predecir Grupo de un Cliente")
    input_df = user_input_features()

    # Modelo por defecto (puedes personalizar la ruta)
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
