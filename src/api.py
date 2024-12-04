import os
import pickle
import uvicorn
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import create_engine, Column, Integer, Float, DateTime, JSON, func, text
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MySQL Configuration
MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
MYSQL_USER = os.getenv('MYSQL_USER', 'root')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'customer_predictions')
MYSQL_PORT = os.getenv('MYSQL_PORT', '3306')

def create_database_if_not_exists():
    """
    Create MySQL database if it doesn't exist using a temporary connection
    """
    try:
        # Temporary connection without specifying database
        temp_engine = create_engine(
            f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}"
        )
        
        with temp_engine.connect() as connection:
            # Use text() to create an executable SQL statement
            connection.execute(text(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}`"))
            connection.commit()
        
        print(f"Database {MYSQL_DATABASE} created or already exists")
    except Exception as e:
        print(f"Error creating database: {e}")
        raise

# Construir la URL de conexión de MySQL
DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"

# Create database before creating engine
create_database_if_not_exists()

# Crear motor de base de datos
engine = create_engine(
    DATABASE_URL, 
    pool_pre_ping=True,  # Verificar conexión antes de usar
    pool_recycle=3600,   # Reciclar conexiones cada hora
    pool_size=10,        # Tamaño del pool de conexiones
    max_overflow=20      # Conexiones adicionales permitidas
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Resto del código permanece igual que en el script original
# Modelo de base de datos para predicciones
class PredictionRecord(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    region = Column(Integer, nullable=False)
    tenure = Column(Integer, nullable=False)
    age = Column(Integer, nullable=False)
    marital = Column(Integer, nullable=False)
    address = Column(Integer, nullable=False)
    income = Column(Float, nullable=False)
    ed = Column(Integer, nullable=False)
    employ = Column(Integer, nullable=False)
    retire = Column(Integer, nullable=False)
    gender = Column(Integer, nullable=False)
    reside = Column(Integer, nullable=False)
    income_per_year_employed = Column(Float, nullable=False)
    age_category = Column(Integer, nullable=False)
    income_age_ratio = Column(Float, nullable=False)
    predicted_group = Column(Integer, nullable=False)
    prediction_probability = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# Modelo de entrada para validación
class CustomerInput(BaseModel):
    region: int = Field(..., ge=1, le=3)
    tenure: int = Field(..., ge=1, le=100)
    age: int = Field(..., ge=1, le=80)
    marital: int = Field(..., ge=0, le=1)
    address: int = Field(..., ge=0, le=60)
    income: float
    ed: int = Field(..., ge=1, le=5)
    employ: int = Field(..., ge=1, le=5)
    retire: int = Field(..., ge=0, le=1)
    gender: int = Field(..., ge=0, le=1)
    reside: int = Field(..., ge=1, le=8)
    income_per_year_employed: float
    age_category: int = Field(..., ge=1, le=4)
    income_age_ratio: float
    
    @field_validator('income', 'income_per_year_employed', 'income_age_ratio')
    def validate_positive_values(cls, v):
        if v < 0:
            raise ValueError('Los valores de ingreso deben ser positivos')
        return v

# Cargar modelo (asegúrate de que la ruta sea correcta)
def load_ml_model(model_path: str = "../src/models/best_model_catboost.pkl"):
    try:
        with open(model_path, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Modelo no encontrado")

# Dependencia para obtener sesión de base de datos
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Crear aplicación FastAPI
app = FastAPI(
    title="Customer Classification API",
    description="API para predecir el grupo de un cliente con MySQL",
    version="1.0.0"
)

@app.get("/")
async def root():
    """
    Endpoint raíz de la API.
    """
    return {"message": "Bienvenido a la API de clasificación de clientes"}

@app.post("/predict/")
async def predict_customer_group(
    customer_data: CustomerInput, 
    db: Session = Depends(get_db)
):
    """
    Endpoint para predecir el grupo de clientes.
    """
    try:
        # Cargar modelo
        model = load_ml_model()
        
        # Convertir datos a DataFrame
        input_df = pd.DataFrame([customer_data.dict()])
        
        # Validar entrada
        if input_df.isnull().values.any():
            raise HTTPException(status_code=422, detail="Faltan datos en la solicitud.")
        
        # Realizar predicción
        prediction = model.predict(input_df)
        probabilities = model.predict_proba(input_df)
        
        # Obtener grupo predicho y probabilidad
        predicted_group = int(prediction[0])
        predicted_probability = float(probabilities[0][predicted_group])
        
        # Almacenar predicción en base de datos
        prediction_record = PredictionRecord(
            **customer_data.dict(),
            predicted_group=predicted_group,
            prediction_probability=predicted_probability
        )
        db.add(prediction_record)
        db.commit()
        db.refresh(prediction_record)
        
        return {
            "predicted_group": predicted_group,
            "probability": predicted_probability,
            "prediction_id": prediction_record.id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando predicción: {str(e)}")

@app.get("/predictions/")
async def get_predictions(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db)
):
    try:
        predictions = db.query(PredictionRecord).offset(skip).limit(limit).all()
        return predictions
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/prediction/{prediction_id}")
async def get_prediction_by_id(
    prediction_id: int, 
    db: Session = Depends(get_db)
):
    prediction = db.query(PredictionRecord).filter(PredictionRecord.id == prediction_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Predicción no encontrada")
    return prediction

@app.get("/model-performance/")
async def get_model_performance(db: Session = Depends(get_db)):
    try:
        # Obtener número total de predicciones
        total_predictions = db.query(PredictionRecord).count()
        
        # Distribución de predicciones por grupo
        group_distribution = db.query(
            PredictionRecord.predicted_group, 
            func.count(PredictionRecord.id).label('count')
        ).group_by(PredictionRecord.predicted_group).all()
        
        return {
            "total_predictions": total_predictions,
            "group_distribution": dict(group_distribution)
        }
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))

# Script de configuración inicial de base de datos
def create_database():
    """
    Crear base de datos MySQL si no existe
    """
    # Conexión sin base de datos específica
    engine_temp = create_engine(
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}"
    )
    
    try:
        with engine_temp.connect() as conn:
            conn.execute("commit")
            conn.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DATABASE}")
        print(f"Base de datos {MYSQL_DATABASE} creada exitosamente")
    except Exception as e:
        print(f"Error al crear base de datos: {e}")

# Ejecutar la API
if __name__ == "__main__":

    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Iniciar servidor
    uvicorn.run(app, host="127.0.0.1", port=8000)
'''`Dependencias adicionales:
```bash
pip install fastapi uvicorn sqlalchemy pymysql python-dotenv mysqlclient pandas numpy scikit-learn
```

Cambios principales:
1. Uso de MySQL con SQLAlchemy
2. URL de conexión configurable via `.env`
3. Modelo de base de datos más detallado
4. Función para crear base de datos si no existe
5. Configuración de pool de conexiones

Antes de ejecutar, asegúrate de:
- Tener MySQL instalado
- Crear un usuario con permisos
- Instalar las dependencias
'''