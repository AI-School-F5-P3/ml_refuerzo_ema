# Proyecto de Clasificación de Clientes

## Descripción
Este proyecto utiliza datos demográficos para predecir la pertenencia de un cliente a uno de cuatro grupos definidos. La solución incluye un modelo de clasificación, seguimiento con MLflow, una API para predicciones y un entorno contenedorizado con Docker.

## Estructura del Proyecto
- **data/**: Datos originales y procesados.
- **src/**: Código fuente, incluyendo modelos, API y utilidades.
- **mlruns/**: Directorio de experimentos MLflow.
- **docker/**: Archivos para contenedorización.
- **metrics/**: Reportes de rendimiento.

## Configuración
### 1. Requisitos
- Python 3.9
- Docker y Docker Compose

### 2. Instalación
#### Sin Docker
1. Crear un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. Configurar la base de datos en .env.

3. Ejecutar:

- **API**:
   uvicorn src.api.api:app --reload

- **Panel de control Streamlit** :
   streamlit run src/app_kpi_ov.py

#### Con Docker
1. Construir y levantar contenedores:
   docker-compose up --build

2. Acceder:
- **API rápida**: http ://localhost :8000
- **Flujo de ml** : http ://localhost :5000

## **Funcionalidades**
**API**
- predicción : Enviar datos demográficos y recibir predicciones.
- Consultas : Acceso al historial de predicciones almacenado en MySQL.

**Seguimiento de MLflow**
- Logueo de métricas de los modelos.
- Almacenamiento de modelos y artefactos.

