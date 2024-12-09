FROM python:3.12-slim

# Instala las herramientas necesarias para compilar y trabajar con MySQL/MariaDB
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libmariadb-dev \
    libmariadb-dev-compat \
    mariadb-client \
    pkg-config \
    libpq-dev \  
    && apt-get clean

# Definir variables de entorno
ENV MYSQLCLIENT_CFLAGS="-I/usr/include/mysql"
ENV MYSQLCLIENT_LDFLAGS="-L/usr/lib/x86_64-linux-gnu -lmysqlclient"

# Actualizar pip
RUN pip install --upgrade pip

# Crear el directorio de trabajo
WORKDIR /app

# Copiar los archivos necesarios
COPY requirements.txt .

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . /app

# Exponer puertos para FastAPI y MLflow
EXPOSE 8000 5000

# Comando por defecto
CMD ["bash", "-c", "mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root /app/mlruns & uvicorn src.api.api:app --host 0.0.0.0 --port 8000"]
