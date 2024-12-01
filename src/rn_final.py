import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
# The KerasClassifier was moved to scikeras in TensorFlow 2.
from scikeras.wrappers import KerasClassifier  
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Asegurarse de que los directorios existen
os.makedirs('../src/models', exist_ok=True)
os.makedirs('../src/metrics', exist_ok=True)
    
def create_model(layers=[256, 128, 64],  # Aumentar complejidad de la red
                dropout_rates=[0.5, 0.4, 0.3],  # Ajustar dropout
                learning_rate=0.0005,  # Reducir learning rate
                l2_lambda=0.0005,  # Ajustar regularización
                input_dim=None,
                num_classes=4):
    model = Sequential()
    
    # Capa de entrada con batch normalization
    model.add(Dense(layers[0], input_dim=input_dim, activation='relu', 
                    kernel_regularizer=l2(l2_lambda)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Dropout(dropout_rates[0]))
    
    # Capas ocultas con batch normalization
    for neurons, dropout_rate in zip(layers[1:], dropout_rates[1:]):
        model.add(Dense(neurons, activation='relu', 
                        kernel_regularizer=l2(l2_lambda)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
    
    return model

def detect_overfitting(history):
    """
    Calcula y imprime métricas de overfitting
    
    Args:
    - history: Historial de entrenamiento del modelo
    
    Returns:
    - Diccionario con métricas de overfitting
    """
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    
    # Calcular el overfitting
    overfitting_loss = train_loss[-1] - val_loss[-1]
    overfitting_accuracy = train_accuracy[-1] - val_accuracy[-1]
    print(f"\nOverfitting - Loss: {overfitting_loss:.4f}, Accuracy: {overfitting_accuracy:.4f}")

    # Calcular brechas entre entrenamiento y validación
    loss_gap = [t - v for t, v in zip(train_loss, val_loss)]
    accuracy_gap = [t - v for t, v in zip(train_accuracy, val_accuracy)]
    
    # Métricas de overfitting
    overfitting_metrics = {
        'final_train_loss': train_loss[-1],
        'final_val_loss': val_loss[-1],
        'final_train_accuracy': train_accuracy[-1],
        'final_val_accuracy': val_accuracy[-1],
        'max_loss_gap': max(loss_gap),
        'max_accuracy_gap': max(accuracy_gap),
        'overfitting_loss' : overfitting_loss,
        'overfitting_accuracy': overfitting_accuracy
    }
    
    with open(os.path.join("../src/metrics/rn_overfitting_metrics.json"), "w") as f:
        json.dump(overfitting_metrics, f, indent=4)
        
    # Imprimir métricas de overfitting
    print("\n--- Métricas de Overfitting ---")
    
    for key, value in overfitting_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Visualizar curvas de aprendizaje
    plt.figure(figsize=(12, 5))
    
    # Gráfico de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Curvas de Pérdida')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.savefig(os.path.join("../src/metrics/rn_loss_curves.png"))
    
    # Gráfico de precisión
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.title('Curvas de Precisión')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    plt.savefig(os.path.join("../src/metrics/rn_accuracy_curves.png"))
    plt.tight_layout()
    plt.show()
    

def stratified_cross_validation(X, y, n_splits=5):
    """
    Perform stratified cross-validation with detailed reporting
    
    Args:
    - X (array): Input features
    - y (array): Target labels
    - n_splits (int): Number of cross-validation splits
    
    Returns:
    - Detailed cross-validation results
    """
    # Preparar el modelo para cross-validation
    model_params = {
        'layers': [256, 128],
        'dropout_rates': [0.5, 0.4],
        'learning_rate': 0.001,
        'l2_lambda': 0.001,
        'input_dim': X.shape[1],
        'num_classes': y.shape[1]
    }

    # Crear KerasClassifier
    # Removed return_train_score=True from KerasClassifier initialization
    keras_classifier = KerasClassifier(
        build_fn=create_model, 
        **model_params,
        epochs=100,
        batch_size=32,
        verbose=0,
        # return_train_score=True  # This line was causing the error
    )
    
    # Configurar validación cruzada estratificada
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Callbacks para cada fold
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=5, 
        min_lr=0.00001
    )
    
    # Contenedores para resultados
    fold_scores = []
    fold_histories = []
    
    # Realizar validación cruzada
    for fold, (train_index, val_index) in enumerate(skf.split(X, np.argmax(y, axis=1)), 1):
        print(f"\n--- Fold {fold} ---")
        
        # Dividir datos
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]
        
        # Entrenar modelo
        history = keras_classifier.fit(
            X_train_fold, y_train_fold, 
            validation_data=(X_val_fold, y_val_fold),
            callbacks=[early_stopping, reduce_lr]
        )
        
        # Evaluar modelo
        # The model is accessed using keras_classifier.model_ 
        val_loss, val_accuracy = keras_classifier.model_.evaluate(X_val_fold, y_val_fold, verbose=0)  
        fold_scores.append(val_accuracy)
        fold_histories.append(history)
        
        print(f"Fold {fold} - Validation Accuracy: {val_accuracy:.4f}")
    
    # Resumen de resultados
    print("\n--- Resumen de Validación Cruzada ---")
    print(f"Precisiones de cada fold: {fold_scores}")
    print(f"Precisión promedio: {np.mean(fold_scores):.4f}")
    print(f"Desviación estándar: {np.std(fold_scores):.4f}")

    return {
        'fold_scores': fold_scores,
        'mean_accuracy': np.mean(fold_scores),
        'std_accuracy': np.std(fold_scores),
        'histories': fold_histories
    }


# Carga de datos
df = pd.read_csv("../data/teleCust1000t.csv")
X = df.drop('custcat', axis=1)
y = df['custcat']

# Preprocesamiento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))

# Realizar validación cruzada
cv_results = stratified_cross_validation(X_scaled, y_encoded)

# Visualización de precisiones por fold
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(cv_results['fold_scores'])+1), cv_results['fold_scores'])
plt.title('Precisión por Fold de Validación Cruzada')
plt.xlabel('Fold')
plt.ylabel('Precisión')
plt.ylim(0, 1)
plt.show()

# Opcional: Entrenamiento final en todo el conjunto de datos
final_model_params = {
    'layers': [256, 128],
    'dropout_rates': [0.5, 0.4],
    'learning_rate': 0.001,
    'l2_lambda': 0.001,
    'input_dim': X_scaled.shape[1],
    'num_classes': y_encoded.shape[1]
}

final_model = create_model(**final_model_params)

# Modificar el entrenamiento final para incluir detección de overfitting
final_history = final_model.fit(
    X_scaled, y_encoded, 
    epochs=150,  # Aumentar épocas
    validation_split=0.2,
    callbacks=[
        EarlyStopping(
            monitor='val_loss', 
            patience=15,  # Aumentar paciencia
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.1,  # Reducir factor de learning rate
            patience=8, 
            min_lr=0.000001
        )
    ]
)

# Detectar overfitting
overfitting_results = detect_overfitting(final_history)

# Evaluar el modelo final
y_pred_proba = final_model.predict(X_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_encoded, axis=1)

# Imprimir informe de clasificación
print("\n--- Métricas del Modelo Final ---")
report = classification_report(y_true, y_pred, target_names=[str(i) for i in encoder.categories_[0]])
print(report)

# Guardar reporte en un archivo
with open(os.path.join("../src/metrics/rn_classification_report.txt"), "w") as f:
    f.write(report)

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
print("\nMatriz de Confusión:")
print(cm)

# Visualizar la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=encoder.categories_[0], 
            yticklabels=encoder.categories_[0])
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Verdadero')
plt.tight_layout()
plt.savefig('../src/metrics/rn_confusion_matrix.png')
plt.show()

# Guardar modelo final
final_model.save("../src/models/rn_model.keras")
