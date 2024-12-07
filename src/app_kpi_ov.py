import streamlit as st
import requests
import pandas as pd
import os
from PIL import Image
import json
import plotly.graph_objects as go
import plotly.express as px

# Configuración de página ANTES de cualquier otro comando de Streamlit
st.set_page_config(
    page_title="Clasificación de Clientes", 
    page_icon=":bar_chart:",
    layout="wide"
)

# Configuración de la URL base de la API
API_BASE_URL = "http://127.0.0.1:8000"

# Agregar logo en la parte superior de la página
logo_path = "../src/images/logo.png"  # Ruta al logo
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.image(logo, width=100)  # Ajusta el tamaño del logo según sea necesario
    
# Título de la aplicación
st.title("Clasificación de Clientes")

# Función para cargar el reporte de clasificación
def load_classification_report(filename="../src/metrics/classification_report.json"):
    """
    Carga el reporte de clasificación desde JSON
    
    Args:
    - filename (str): Ruta del archivo JSON
    
    Returns:
    - dict: Reporte de clasificación
    """
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def load_metrics_file(filename="../src/metrics/rn_metrics.json"):
    """
    Carga métricas adicionales desde JSON
    
    Args:
    - filename (str): Ruta del archivo JSON
    
    Returns:
    - dict: Métricas adicionales
    """
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"No se encontró el archivo de métricas: {filename}")
        return None

def load_overfitting_metrics(filename="../src/metrics/rn_overfitting_metrics.json"):
    """
    Carga las métricas de overfitting desde un archivo JSON.

    Args:
    - filename (str): Ruta del archivo JSON.

    Returns:
    - dict: Métricas de overfitting.
    """
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"No se encontró el archivo de métricas de overfitting: {filename}")
        return None

def create_kpi_cards(metrics):
    """
    Create KPI cards for model performance metrics
    """
    cols = st.columns(4)
    kpi_mapping = {
        'Accuracy': metrics.get('Accuracy', 0),
        'Macro Precision': metrics.get('Macro Precision', 0),
        'Macro Recall': metrics.get('Macro Recall', 0),
        'Macro F1-Score': metrics.get('Macro F1-Score', 0)
    }
    
    for i, (metric_name, metric_value) in enumerate(kpi_mapping.items()):
        with cols[i]:
            st.metric(label=metric_name, value=f"{metric_value:.4f}")

def plot_performance_radar(metrics):
    """
    Create a radar chart for model performance metrics
    """
    categories = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1-Score']
    values = [
        metrics.get('Accuracy', 0),
        metrics.get('Macro Precision', 0),
        metrics.get('Macro Recall', 0),
        metrics.get('Macro F1-Score', 0)
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself'
    ))
    
    fig.update_layout(
        title='Model Performance Radar Chart',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False
    )
    
    st.plotly_chart(fig)

def display_class_performance(classification_report):
    """
    Display performance metrics for each class
    """
    class_metrics = {
        'Class': list(classification_report.keys())[:-3],
        'Precision': [class_data['precision'] for class_data in list(classification_report.values())[:-3]],
        'Recall': [class_data['recall'] for class_data in list(classification_report.values())[:-3]],
        'F1-Score': [class_data['f1-score'] for class_data in list(classification_report.values())[:-3]]
    }
    
    df = pd.DataFrame(class_metrics)
    st.dataframe(df)

def load_metrics_and_reports():
    """
    Load metrics and classification reports from various possible file locations
    """
    metrics_files = [
        "../src/metrics/rn_metrics.json",
        "../src/metrics/rn_classification.json",
        "../src/metrics/classification_report.json"
    ]
    
    classification_files = [
        "../src/metrics/classification_report.json", 
        "../src/metrics/rn_classification_report.json"
    ]
    
    metrics = None
    classification_report = None
    
    for file in metrics_files:
        try:
            with open(file, 'r') as f:
                metrics = json.load(f)
                break
        except FileNotFoundError:
            continue
    
    for file in classification_files:
        try:
            with open(file, 'r') as f:
                classification_report = json.load(f)
                break
        except FileNotFoundError:
            continue
    
    return metrics, classification_report

def show_performance_metrics():
    """
    Comprehensive performance metrics section
    """
    st.markdown("## Informe de Rendimiento del Modelo RN")
    
    metrics, classification_report = load_metrics_and_reports()
    
    if metrics is None or classification_report is None:
        st.error("No se encontraron archivos de métricas o reportes de clasificación.")
        return
    
    # KPI Cards
    st.markdown("### Métricas Principales")
    create_kpi_cards(metrics)
    
    # Performance Radar Chart
    st.markdown("### Radar de Rendimiento del Modelo")
    plot_performance_radar(metrics)
    
    # Per Class Performance
    st.markdown("### Rendimiento por Clase")
    display_class_performance(classification_report)
    
    # Model Visualizations
    st.markdown("### Visualizaciones del Modelo")
    
    metrics_dir = "../src/metrics"
    rn_graphs = [
        os.path.join(metrics_dir, file) 
        for file in os.listdir(metrics_dir) 
        if file.startswith("rn_") and file.endswith(".png")
    ]
    
    if rn_graphs:
        cols = st.columns(len(rn_graphs))
        for i, graph_path in enumerate(rn_graphs):
            with cols[i]:
                st.image(graph_path, caption=os.path.basename(graph_path), use_container_width=True)
    else:
        st.error("No se encontraron gráficas para el modelo RN.")    
        
# Función para ingresar los datos de un cliente (en tres columnas)
def user_input_features():
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Edad", min_value=18, max_value=80, step=1)
        gender = st.selectbox("Género", [0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
        marital = st.selectbox("Estado Civil", [0, 1], format_func=lambda x: "No Casado" if x == 0 else "Casado")
        ed = st.slider("Nivel Educativo (1 a 5)", min_value=1, max_value=5, step=1)
    with col2:
        region = st.slider("Región", min_value=1, max_value=3, step=1)
        reside = st.slider("Nº de personas en el hogar", min_value=1, max_value=8, step=1)
        tenure = st.slider("Tiempo de permanencia (en meses)", min_value=1, max_value=72, step=1)
        address = st.slider("Dirección (años)", min_value=0, max_value=55, step=1)
    with col3:
        income = st.slider("Ingresos", min_value=9, max_value=1700, step=1)
        employ = st.slider("Años trabajando", min_value=1, max_value=47, step=1)
        retire = st.selectbox("¿Está Jubilado?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")

    # Crear diccionario con los datos ingresados
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
    }
    return data

# Menú lateral
menu = st.sidebar.selectbox(
    "Selecciona una opción",
    [
        "Predecir Grupo de un Cliente",
        "Informe de Rendimiento", 
        "Comparación del Modelo"
    ]
)

# Lógica de las opciones del menú
if menu == "Predecir Grupo de un Cliente":
    st.markdown("## Predecir Grupo de un Cliente")
    
    # Capturar datos del cliente
    input_data = user_input_features()
    
    if st.button("Predecir Grupo"):
        try:
            # Enviar solicitud a la API
            response = requests.post(f"{API_BASE_URL}/predict/", json=input_data)
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"El cliente pertenece al grupo: {result['predicted_group']}")
                st.write(f"Probabilidad de pertenecer al grupo {result['predicted_group']}: {result['probability']:.2%}")
                st.write(f"ID de predicción: {result.get('prediction_id')}")
            else:
                st.error(f"Error en la predicción: {response.text}")
        
        except requests.exceptions.RequestException as e:
            st.error(f"Error de conexión con la API: {e}")

    # Mostrar historial de predicciones
    st.markdown("## Historial de Predicciones")
    try:
        # Obtener predicciones de la API
        response = requests.get(f"{API_BASE_URL}/predictions/?skip=0&limit=10")
        
        if response.status_code == 200:
            predictions = response.json()
            if predictions:
                st.dataframe(predictions)
            else:
                st.warning("No hay predicciones disponibles")
        else:
            st.error(f"Error al cargar predicciones: {response.text}")
    
    except requests.exceptions.RequestException as e:
        st.error(f"Error de conexión con la API: {e}")

elif menu == "Informe de Rendimiento":
    #show_performance_metrics()
        # Crear pestañas
    tab1, tab2, tab3, tab4 = st.tabs(["Métricas Principales", "Radar de Rendimiento", "Rendimiento por Clase", "Visualizaciones"])
    
    metrics, classification_report = load_metrics_and_reports()
    overfitting_metrics = load_overfitting_metrics()
    
    if metrics is None or classification_report is None or overfitting_metrics is None:
        st.error("No se encontraron algunos archivos necesarios para las métricas.")
    else:
        # Contenido de la pestaña "Métricas Principales"
        with tab1:
            st.markdown("### Métricas Principales")
            
            # Agregar las métricas de overfitting
            overfitting_value = overfitting_metrics.get('Overfitting', None)
            if overfitting_value is not None:
                metrics['Overfitting'] = overfitting_value
            
            # Crear KPI cards, incluyendo overfitting
            cols = st.columns(5)  # Aumentar el número de columnas para incluir overfitting
            kpi_mapping = {
                'Accuracy': metrics.get('Accuracy', 0),
                'Macro Precision': metrics.get('Macro Precision', 0),
                'Macro Recall': metrics.get('Macro Recall', 0),
                'Macro F1-Score': metrics.get('Macro F1-Score', 0),
                'Overfitting': metrics.get('Overfitting', 0)
            }
            
            for i, (metric_name, metric_value) in enumerate(kpi_mapping.items()):
                with cols[i]:
                    st.metric(label=metric_name, value=f"{metric_value:.4f}" if metric_name != "Overfitting" else f"{metric_value:.2%}")
        
        # Contenido de la pestaña "Radar de Rendimiento"
        with tab2:
            st.markdown("### Radar de Rendimiento del Modelo")
            plot_performance_radar(metrics)
        
        # Contenido de la pestaña "Rendimiento por Clase"
        with tab3:
            st.markdown("### Rendimiento por Clase")
            display_class_performance(classification_report)
        
        # Contenido de la pestaña "Visualizaciones"
        with tab4:
            st.markdown("### Visualizaciones del Modelo")
            
            metrics_dir = "../src/metrics"
            rn_graphs = [
                os.path.join(metrics_dir, file) 
                for file in os.listdir(metrics_dir) 
                if file.startswith("rn_") and file.endswith(".png")
            ]
            
            if rn_graphs:
                # Mostrar las gráficas en 2 filas y 2 columnas
                for i in range(0, len(rn_graphs), 2):  # Iterar en bloques de 2
                    cols = st.columns(2)
                    for j, col in enumerate(cols):
                        if i + j < len(rn_graphs):  # Asegurarse de que no se salga de rango
                            with col:
                                st.image(rn_graphs[i + j], use_container_width=True)
            else:
                st.error("No se encontraron gráficas para el modelo RN.")

elif menu == "Comparación del Modelo":
    st.markdown("## Comparación de Modelos")
    
    # Obtener todos los modelos (excluyendo RN)
    metrics_dir = "../src/metrics"
    models = set(
        file.split('_')[0].lower() 
        for file in os.listdir(metrics_dir) 
        if not file.startswith("rn_") and file.endswith(".png")
    )
    
    comparison_file = "../src/metrics/model_comparison.csv"
    performance = os.path.join(metrics_dir, "model_performance_comparison.png")

    try:
        comparison_df = pd.read_csv(comparison_file)
        
        # Mostrar la gráfica general de comparación de modelos
        if os.path.exists(performance):
            st.image(performance, use_container_width=True, caption="Comparación de Modelos")
        else:
            st.warning("No se encontró la gráfica 'model_performance_comparison.png' en el directorio especificado.")
    
        # Verificar las columnas disponibles
        st.write(comparison_df)

        # Determinar el nombre correcto de la columna de modelos
        column_name = 'Model' if 'Model' in comparison_df.columns else 'Model' if 'Model' in comparison_df.columns else None
        if column_name is None:
            st.error("No se encontró una columna adecuada para identificar los modelos ('Model' o 'Model').")
            
        
        # Filtrar los modelos para incluir solo los que están en el archivo de comparación
        available_models = set(comparison_df[column_name].str.lower())
        valid_models = models.intersection(available_models)
        
        if not valid_models:
            st.warning("No hay modelos válidos disponibles para la comparación.")
            
        
        # Crear pestañas para los modelos válidos
        tabs = st.tabs([model.upper() for model in valid_models])
        
        for i, model in enumerate(valid_models):
            with tabs[i]:
                st.markdown(f"### Gráficas del Modelo {model.upper()}")
                
                # Mostrar gráficas del modelo
                model_graphs = [
                    os.path.join(metrics_dir, file) 
                    for file in os.listdir(metrics_dir) 
                    if file.startswith(f"{model}_") and file.endswith(".png")
                ]
                
                if model_graphs:
                    cols = st.columns(len(model_graphs))
                    for j, graph_path in enumerate(model_graphs):
                        with cols[j]:
                            st.image(graph_path, use_container_width=True)
                else:
                    st.error(f"No se encontraron gráficas para el modelo {model.upper()}.")
                
                st.markdown(f"### Métricas del Modelo {model.upper()}")
                
                # Filtrar métricas para el modelo actual
                model_metrics = comparison_df[comparison_df[column_name].str.lower() == model]
                
                if not model_metrics.empty:
                    # Eliminar columnas innecesarias si es necesario
                    model_metrics = model_metrics.drop(columns=['ColumnaInnecesaria1', 'ColumnaInnecesaria2'], errors='ignore')
                    st.dataframe(model_metrics)
                else:
                    st.warning(f"No se encontraron métricas para el modelo {model.upper()} en el archivo de comparación.")
    
    except FileNotFoundError:
        st.error(f"No se encontró el archivo '{comparison_file}'.")