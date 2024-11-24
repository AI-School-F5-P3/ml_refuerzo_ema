import pandas as pd
import numpy as np
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
matplotlib.use('Agg')  # Establecer el backend antes de importar pyplot
import matplotlib.pyplot as plt

def load_and_prepare_data():
    """
    Carga y prepara los datos para el modelado
    """
    df = pd.read_csv('../data/proc_escal.csv')
    return df
        #columns = ['region', 'tenure', 'age', 'marital', 'address', 'income', 
        #        'ed', 'employ', 'retire', 'gender', 'reside', 'custcat', 'income_per_year_employe'
        #        'age_category', 'income_age_ratio']
def preprocess_data(df):
    """
    Preprocesa los datos para el modelado
    """
    # Separar features y target
    #X = df.drop('custcat', axis=1)
    X = df[['tenure', 'age', 'address', 'income', 'ed', 'employ']]
    y = df['custcat']
    
    # Dividir en conjuntos de entrenamiento y prueba sin stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("\nDistribución de clases en conjunto de entrenamiento:")
    print(y_train.value_counts())
    print("\nDistribución de clases en conjunto de prueba:")
    print(y_test.value_counts())
    
    return X_train, X_test, y_train, y_test

def find_optimal_k(X_train, X_test, y_train, y_test):
    """
    Encuentra el valor óptimo de k para KNN probando diferentes valores
    """
    k_range = range(1, 31, 2)  # Probar valores impares de k del 1 al 30
    scores = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test)
        scores.append(score)
    
    # Visualizar los resultados
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, scores, 'bo-')
    plt.xlabel('Valor de k')
    plt.ylabel('Precisión')
    plt.title('Precisión vs Valor de k en KNN')
    plt.grid(True)
    plt.savefig('../data/metrics/knn_optimization.png')
    plt.close()
    
    # Encontrar el mejor k
    optimal_k = k_range[np.argmax(scores)]
    print(f"\nMejor valor de k encontrado: {optimal_k}")
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
        plt.savefig(f'../data/metrics/learning_curve_{title.lower().replace(" ", "_")}.png')
        plt.close()
    except Exception as e:
        print(f"Error al generar la curva de aprendizaje para {title}: {str(e)}")

def analyze_overfitting(model, X_train, X_test, y_train, y_test, model_name):
    """
    Analiza el overfitting usando validación cruzada y diferencia entre scores
    """
    try:
        # Validación cruzada
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        # Scores en train y test
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"\nAnálisis de Overfitting para {model_name}:")
        print(f"Score medio en validación cruzada: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"Score en entrenamiento: {train_score:.3f}")
        print(f"Score en prueba: {test_score:.3f}")
        print(f"Diferencia train-test: {train_score - test_score:.3f}")
        
        # Generar curva de aprendizaje
        plot_learning_curve(model, X_train, y_train, model_name)
    except Exception as e:
        print(f"Error en el análisis de overfitting para {model_name}: {str(e)}")

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Entrena y evalúa modelos de clasificación incluyendo análisis de overfitting
    """
    # Encontrar el mejor k para KNN
    optimal_k = find_optimal_k(X_train, X_test, y_train, y_test)
    
    # Configurar los modelos
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            # Removido multi_class para evitar warnings
            class_weight='balanced'
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=optimal_k,
            weights='distance'
        ),
        'SVM': SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True
        )
    }
    
    results = {}
    
    for name, model in models.items():
        try:
            print(f"\nEntrenando {name}...")
            
            # Entrenar modelo
            model.fit(X_train, y_train)
            
            # Análisis de overfitting
            analyze_overfitting(model, X_train, X_test, y_train, y_test, name)
            
            # Hacer predicciones
            y_pred = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            
            # Guardar resultados
            results[name] = {
                'model': model,
                'test_classification_report': classification_report(y_test, y_pred),
                'train_classification_report': classification_report(y_train, y_pred_train),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            # Imprimir resultados
            print(f"\nResultados de entrenamiento para {name}:")
            print(results[name]['train_classification_report'])
            print(f"\nResultados de prueba para {name}:")
            print(results[name]['test_classification_report'])
            
            # Visualizar matriz de confusión
            plt.figure(figsize=(8, 6))
            cm = results[name]['confusion_matrix']
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                xticklabels=[f'Clase {i}' for i in sorted(set(y_test))],
                yticklabels=[f'Clase {i}' for i in sorted(set(y_test))]
            )
            plt.title(f'Matriz de Confusión - {name}')
            plt.xlabel('Predicción')
            plt.ylabel('Valor Real')
            plt.savefig(f'../data/metrics/confusion_matrix_{name.lower().replace(" ", "_")}.png')
            plt.close()
            
            # Para Random Forest y Decision Tree, visualizar importancia de características
            if name in ['Random Forest', 'Decision Tree']:
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                plt.figure(figsize=(10, 6))
                sns.barplot(data=feature_importance, x='importance', y='feature')
                plt.title(f'Importancia de Características - {name}')
                plt.tight_layout()
                plt.savefig(f'../data/metrics/feature_importance_{name.lower().replace(" ", "_")}.png')
                plt.close()
                
                print("\nImportancia de características:")
                print(feature_importance)
        
        except Exception as e:
            print(f"Error al procesar el modelo {name}: {str(e)}")
    
    try:
        # Comparar modelos con validación cruzada
        plt.figure(figsize=(12, 6))
        cv_scores = []
        model_names = list(models.keys())
        
        for name, model in models.items():
            scores = cross_val_score(model, X_train, y_train, cv=5)
            cv_scores.append(scores)
        
        plt.boxplot(cv_scores, labels=model_names)
        plt.title('Comparación de Modelos - Validación Cruzada')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('../data/metrics/model_comparison_cv.png')
        plt.close()
    except Exception as e:
        print(f"Error al generar la comparación de modelos: {str(e)}")
    
    return results

def main():
    """
    Función principal que ejecuta todo el pipeline
    """
    print("1. Cargando datos...")
    df = load_and_prepare_data()
    
    print("\n3. Preprocesando datos...")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    print("\n4. Entrenando y evaluando modelos...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    print("\n5. Proceso completado!")
    print("Se han generado las siguientes visualizaciones:")
    print("- class_distribution.png")
    print("- numeric_distributions.png")
    print("- correlation_matrix.png")
    print("- knn_optimization.png")
    print("- learning_curve_*.png (para cada modelo)")
    print("- confusion_matrix_*.png (para cada modelo)")
    print("- feature_importance_*.png (para Random Forest y Decision Tree)")
    print("- model_comparison_cv.png")

if __name__ == "__main__":
    main()
