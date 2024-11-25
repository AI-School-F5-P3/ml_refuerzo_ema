import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# Asegurarse de que los directorios existen
os.makedirs('../src/models', exist_ok=True)
os.makedirs('../src/metrics', exist_ok=True)

def load_and_prepare_data():
    """
    Carga y prepara los datos para el modelado
    """
    df = pd.read_csv('../data/proc_escal.csv')
    return df

def preprocess_data(df):
    """
    Preprocesa los datos para el modelado
    """
    #X = df[['tenure', 'age', 'address', 'income', 'ed', 'employ']]
    X = df.drop('custcat', axis = 1)
    y = df['custcat']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def find_optimal_k(X_train, X_test, y_train, y_test):
    """
    Encuentra el valor óptimo de k para KNN probando diferentes valores
    """
    k_range = range(1, 31, 2)
    scores = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    
    # Visualizar los resultados
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, scores, 'bo-')
    plt.xlabel('Valor de k')
    plt.ylabel('Precisión')
    plt.title('Precisión vs Valor de k en KNN')
    plt.grid(True)
    plt.savefig('../src/metrics/knn_optimization.png')
    plt.close()
    
    optimal_k = k_range[np.argmax(scores)]
    return optimal_k

def plot_learning_curve(estimator, X, y, title):
    """
    Genera y visualiza las curvas de aprendizaje para analizar el overfitting
    """
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
        plt.plot(train_sizes, test_mean, label='Cross-validation score', color='green', marker='o')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')
        
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title(f'Learning Curve - {title}')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(f'../src/metrics/learning_curve_{title.lower().replace(" ", "_")}.png')
        plt.close()
    except Exception as e:
        print(f"Error al generar la curva de aprendizaje para {title}: {str(e)}")

def analyze_overfitting(model, X_train, X_test, y_train, y_test, model_name):
    """
    Analiza el overfitting usando validación cruzada y diferencia entre scores
    """
    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"\nAnálisis de Overfitting para {model_name}:")
        print(f"Score medio en validación cruzada: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"Score en entrenamiento: {train_score:.3f}")
        print(f"Score en prueba: {test_score:.3f}")
        print(f"Diferencia train-test: {train_score - test_score:.3f}")
        
        plot_learning_curve(model, X_train, y_train, model_name)
    except Exception as e:
        print(f"Error en el análisis de overfitting para {model_name}: {str(e)}")

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Entrena y evalúa modelos de clasificación incluyendo análisis de overfitting
    """
    optimal_k = find_optimal_k(X_train, X_test, y_train, y_test)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'KNN': KNeighborsClassifier(n_neighbors=optimal_k, weights='distance'),
        'SVM': SVC(kernel='rbf', class_weight='balanced', probability=True)
    }
    
    results = []
    
    for name, model in models.items():
        try:
            print(f"\nEntrenando {name}...")
            
            model.fit(X_train, y_train)
            
            analyze_overfitting(model, X_train, X_test, y_train, y_test, name)
            
            y_pred = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            
            # Guardar el modelo en un archivo pkl
            model_filename = f'../src/models/{name.lower().replace(" ", "_")}.pkl'
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)
            
            # Guardar las métricas en el DataFrame
            model_metrics = {
                'Model': name,
                'Train Classification Report': classification_report(y_train, y_pred_train, output_dict=True),
                'Test Classification Report': classification_report(y_test, y_pred, output_dict=True),
                'Confusion Matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            results.append(model_metrics)
            
            # Visualizar la matriz de confusión
            plt.figure(figsize=(8, 6))
            cm = model_metrics['Confusion Matrix']
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=[f'Clase {i}' for i in sorted(set(y_test))], 
                        yticklabels=[f'Clase {i}' for i in sorted(set(y_test))])
            plt.title(f'Matriz de Confusión - {name}')
            plt.xlabel('Predicción')
            plt.ylabel('Valor Real')
            plt.savefig(f'../src/metrics/confusion_matrix_{name.lower().replace(" ", "_")}.png')
            plt.close()
        
        except Exception as e:
            print(f"Error al procesar el modelo {name}: {str(e)}")
    
    # Crear un DataFrame para comparar las métricas de los modelos
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv('../src/metrics/models_comparison.csv', index=False)
    
    return metrics_df

def main():
    """
    Función principal que ejecuta todo el pipeline
    """
    print("1. Cargando datos...")
    df = load_and_prepare_data()
    
    print("\n3. Preprocesando datos...")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    print("\n4. Entrenando y evaluando modelos...")
    metrics_df = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    print("\n5. Proceso completado!")
    print(f"\nLas métricas de los modelos se han guardado en '../src/metrics/models_comparison.csv'.")
    print("Se han generado las siguientes visualizaciones y modelos:")
    print("- learning_curve_*.png (para cada modelo)")
    print("- confusion_matrix_*.png (para cada modelo)")
    print("- Los modelos se han guardado como pkl en '../src/models/'.")

if __name__ == "__main__":
    main()
