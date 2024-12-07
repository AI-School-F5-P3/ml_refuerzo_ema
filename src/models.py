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

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
import catboost as cb

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Asegurarse de que los directorios existen
os.makedirs('../src/models', exist_ok=True)
os.makedirs('../src/metrics', exist_ok=True)

def load_and_prepare_data():
    """
    Carga y prepara los datos para el modelado
    """
    df = pd.read_csv('../data/teleCust1000t.csv')
    return df

def preprocess_data(df):
    """
    Preprocesa los datos para el modelado
    """
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
        
        plt.figure(figsize=(8, 4))
        plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
        plt.plot(train_sizes, test_mean, label='Cross-validation score', color='green', marker='o')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')
        
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title(f'Learning Curve - {title}')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(f'../src/metrics/{title.lower().replace(" ", "_")}_learning_curve_.png')
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
        
        overfitting = train_score - test_score

        
        print(f"\nAnálisis de Overfitting para {model_name}:")
        print(f"Score medio en validación cruzada: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
        print(f"Score en entrenamiento: {train_score:.2f}")
        print(f"Score en prueba: {test_score:.2f}")
        print(f"Overfitting: {overfitting:.2f}")
        
        plot_learning_curve(model, X_train, y_train, model_name)
    except Exception as e:
        print(f"Error en el análisis de overfitting para {model_name}: {str(e)}")

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Entrena y evalúa modelos de clasificación incluyendo análisis de overfitting
    """
    optimal_k = find_optimal_k(X_train, X_test, y_train, y_test)
    
    models = {
        'LogisticRegresion': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'DecisionTree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'KNN': KNeighborsClassifier(n_neighbors=optimal_k, weights='distance'),
        'SVM': SVC(kernel='rbf', class_weight='balanced', probability=True),
        
        # Nuevos modelos
        'GaussianNB': GaussianNB(),
        'MultimodalNB': MultinomialNB(),
        'ComplementNB': ComplementNB(),
        
        # CatBoost
        'CatBoost': cb.CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            random_seed=42,
            loss_function='MultiClass',
            verbose=0
        )        
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
            plt.figure(figsize=(8, 4))
            cm = model_metrics['Confusion Matrix']
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=[f'Clase {i}' for i in sorted(set(y_test))], 
                        yticklabels=[f'Clase {i}' for i in sorted(set(y_test))])
            plt.title(f'Matriz de Confusión - {name}')
            plt.xlabel('Predicción')
            plt.ylabel('Valor Real')
            plt.savefig(f'../src/metrics/{name.lower().replace(" ", "_")}_confusion_matriz.png')
            plt.close()
        
        except Exception as e:
            print(f"Error al procesar el modelo {name}: {str(e)}")
    
    # Crear un DataFrame para comparar las métricas de los modelos
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv('../src/metrics/models_comparison.csv', index=False)

    return metrics_df

def get_model_metrics(metrics_df):
    """
    Extracts and formats desired metrics from the metrics DataFrame.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing model evaluation metrics.

    Returns:
        list: List of lists, where each inner list represents the metrics for a model.
    """
    model_metrics = []
    
    for idx, row in metrics_df.iterrows():
                # Obtener los scores de entrenamiento y prueba para calcular overfitting
        train_score = row['Train Classification Report']['accuracy']
        test_score = row['Test Classification Report']['accuracy']
        
        # Calcular el porcentaje de overfitting
        overfitting = train_score - test_score
        
        report = row["Test Classification Report"]
        model_metrics.append([
        row["Model"],
        report["accuracy"],
        report["macro avg"]["precision"],
        report["macro avg"]["recall"],
        report["macro avg"]["f1-score"],
        overfitting # Porcentaje de overfitting
        ])
    
    return model_metrics

def plot_model_performance(comparison_df):
    """
    Crea gráficos de barras para comparar accuracy y overfitting entre modelos
    
    Args:
        comparison_df (pd.DataFrame): DataFrame con métricas de los modelos
    """
    # Crear una paleta de colores única para cada modelo
    palette = sns.color_palette("viridis", n_colors=len(comparison_df))
    
    # Preparar figure con dos subplots
    plt.figure(figsize=(15, 6))
    
    # Subplot 1: Accuracy
    plt.subplot(1, 2, 1)
    sns.barplot(x="Model", y="Accuracy", data=comparison_df, hue="Model", palette=palette, legend=False)
    plt.title('Accuracy por Modelo')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Subplot 2: Overfitting
    plt.subplot(1, 2, 2)

    # Ordenar el DataFrame de mayor a menor 'Overfitting'
    #comparison_df_sorted = comparison_df.sort_values(by='Overfitting', ascending=False)
    
    # Asegurarse de que la columna 'Overfitting' sea numérica
    comparison_df['Overfitting'] = pd.to_numeric(comparison_df['Overfitting'], errors='coerce')
    
    sns.barplot(x="Model", y="Overfitting", data=comparison_df, hue="Model", palette=palette, legend=False)

    # Asegurar que el eje y empiece en 0
    plt.ylim(0, comparison_df['Overfitting'].max() * 1.1)

    # Graficar el gráfico de Overfitting ordenado
    sns.barplot(x="Model", y="Overfitting", data=comparison_df, hue="Model", palette=palette, legend=False)
    #sns.barplot(x="Model", y="Overfitting", data=comparison_df, hue="Model", palette=palette, legend=False)
    plt.title('Overfitting por Modelo')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Guardar figura
    plt.savefig('../src/metrics/model_performance_comparison.png', bbox_inches='tight')
    plt.close()
    
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

    # Now metrics_df is defined within main
    model_metrics = get_model_metrics(metrics_df)   
    comparison_df = pd.DataFrame(model_metrics, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Overfitting"])
    comparison_df[["Model","Accuracy", "Precision", "Recall", "F1-Score"]] = comparison_df[["Model","Accuracy", "Precision", "Recall", "F1-Score"]].round(2)
    comparison_df["Overfitting"] = comparison_df["Overfitting"].apply(lambda x: f"{x:.2f}")
    
    # Print or save the comparison DataFrame
    print(comparison_df.to_string())  # Print the DataFrame as a string
    comparison_df.to_csv("../src/metrics/model_comparison.csv", index=False)  # Save to a CSV file
    
    # Después de crear comparison_df
    plot_model_performance(comparison_df)
    
    print("\n5. Proceso completado!")
    print(f"\nLas métricas de los modelos se han guardado en '../src/metrics/models_comparison.csv'.")
    print("Se han generado las siguientes visualizaciones y modelos:")
    print("- learning_curve_*.png (para cada modelo)")
    print("- confusion_matrix_*.png (para cada modelo)")
    print("- Los modelos se han guardado como pkl en '../src/models/'.")

if __name__ == "__main__":
    main()
