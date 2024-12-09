import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from optuna import create_study
from optuna.samplers import TPESampler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Crear directorios si no existen
os.makedirs('../src/models', exist_ok=True)
os.makedirs('../src/metrics', exist_ok=True)

def load_and_prepare_data():
    """
    Carga y prepara los datos para el modelado
    """
    df = pd.read_csv('../data/processed.csv')
    return df

def preprocess_data(df):

    """
    Preprocesa los datos para el modelado
    """
    X = df.drop('custcat', axis=1)
    y = df['custcat'] - 1  # Reescalar las clases para que comiencen en 0

    print("Clases únicas antes del preprocesamiento:", df['custcat'].unique())
    print("Clases únicas después del preprocesamiento:", y.unique())
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
    return X_train, X_test, y_train, y_test

def objective(trial, X_train, y_train):
    """
    Define la función objetivo para Optuna
    """
    model_type = trial.suggest_categorical("model_type", ["xgb", "lgb", "catboost"])
    
    if model_type == "xgb":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10),
        }
        model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    elif model_type == "lgb":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", -1, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10),
        }
        model = lgb.LGBMClassifier(**params, random_state=42)
    
    elif model_type == "catboost":
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, step=0.01),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        }
        model = CatBoostClassifier(**params, random_state=42, verbose=0)
    
    # Validación cruzada
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(X_tr, y_tr)
        scores.append(model.score(X_val, y_val))
    
    return np.mean(scores)


def train_best_model(X_train, y_train, X_test, y_test):
    """
    Entrena el mejor modelo encontrado por Optuna
    """
    study = create_study(direction="maximize", sampler=TPESampler())
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50)
    
    print("Mejores parámetros encontrados por Optuna:")
    print(study.best_params)
    
    model_type = study.best_params.pop("model_type")

    # Filtra los parámetros según el modelo
    if model_type == "xgb":
        valid_params = {k: study.best_params[k] for k in [
            "n_estimators", "max_depth", "learning_rate", "subsample", 
            "colsample_bytree", "gamma", "scale_pos_weight"
        ]}
        model = xgb.XGBClassifier(**valid_params, use_label_encoder=False, eval_metric='logloss', random_state=42)

    elif model_type == "lgb":
        valid_params = {k: study.best_params[k] for k in [
            "n_estimators", "max_depth", "learning_rate", "num_leaves", 
            "subsample", "colsample_bytree", "scale_pos_weight"
        ]}
        model = lgb.LGBMClassifier(**valid_params, random_state=42)

    elif model_type == "catboost":
        valid_params = {k: study.best_params[k] for k in [
            "iterations", "learning_rate", "depth", "l2_leaf_reg"
        ]}
        model = CatBoostClassifier(**valid_params, random_state=42, verbose=0)

    print("Validando parámetros para el modelo seleccionado...")
    print(f"Tipo de modelo: {model_type}")
    print(f"Parámetros usados: {valid_params}")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluación
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f'Matriz de Confusión - Mejor Modelo ({model_type})')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.savefig('../src/metrics/best_model_confusion_matrix.png')
    plt.close()
    
    # Guardar el modelo
    model_filename = f'../src/models/best_model_{model_type}.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    # Análisis de Overfitting
    print(f"\nAnálisis de Overfitting para {model_type}:")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    overfitting_percentage = ((train_score - test_score) / train_score) * 100

    print(f"Score medio en validación cruzada: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
    print(f"Score en entrenamiento: {train_score:.3f}")
    print(f"Score en prueba: {test_score:.3f}")
    print(f"Overfitting: {overfitting_percentage:.2f}%")

    # Curvas de aprendizaje
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, n_jobs=-1)
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), label='Train Score', color='blue')
    plt.plot(train_sizes, test_scores.mean(axis=1), label='Test Score', color='green')
    plt.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_scores.mean(axis=1) - test_scores.std(axis=1),
                    test_scores.mean(axis=1) + test_scores.std(axis=1), alpha=0.1, color='green')
    plt.title(f'Curvas de Aprendizaje - {model_type}')
    plt.xlabel('Tamaño de Entrenamiento')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'../src/metrics/learning_curve_{model_type}.png')
    plt.close()
    
    return model

def main():
    """
    Función principal que ejecuta todo el pipeline
    """
    print("1. Cargando datos...")
    df = load_and_prepare_data()
    
    print("\n2. Preprocesando datos...")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    print("\n3. Optimizando hiperparámetros y entrenando modelo...")
    best_model = train_best_model(X_train, y_train, X_test, y_test)
    
    print("\nProceso completado!")
    print("El mejor modelo se ha guardado en '../src/models/'.")
    print("La matriz de confusión se encuentra en '../src/metrics/best_model_confusion_matrix.png'.")

if __name__ == "__main__":
    main()
