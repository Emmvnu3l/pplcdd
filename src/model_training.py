"""
Módulo de entrenamiento de modelos para el proyecto de análisis de cáncer de piel.

Contiene funciones para definición, entrenamiento y evaluación de modelos.
"""

import pandas as pd
import numpy as np
import os
import joblib
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report)


def get_models_dict(random_state: int = 42) -> Dict[str, Any]:
    """
    Obtiene un diccionario con los modelos a entrenar.
    
    Args:
        random_state: Semilla para reproducibilidad
        
    Returns:
        Diccionario {nombre_modelo: instancia_modelo}
    """
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=random_state, 
            max_iter=1000,
            class_weight='balanced'  # Manejar desbalance
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=random_state,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            random_state=random_state,
            n_estimators=100,
            class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            random_state=random_state,
            n_estimators=100
        ),
        'SVM': SVC(
            random_state=random_state,
            probability=True,
            class_weight='balanced'
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5
        ),
        'Naive Bayes': GaussianNB()
    }
    
    return models


def train_and_evaluate_model(model, model_name: str, 
                            X_train, X_test, 
                            y_train, y_test) -> Dict[str, float]:
    """
    Entrena un modelo y calcula métricas de evaluación.
    
    Args:
        model: Instancia del modelo
        model_name: Nombre del modelo
        X_train: Features de entrenamiento
        X_test: Features de prueba
        y_train: Target de entrenamiento
        y_test: Target de prueba
        
    Returns:
        Diccionario con métricas
    """
    # Entrenar
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    results = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else 0.5
    }
    
    print(f"\n{model_name} Results:")
    print(f"  Accuracy:  {results['Accuracy']:.4f}")
    print(f"  Precision: {results['Precision']:.4f}")
    print(f"  Recall:    {results['Recall']:.4f}")
    print(f"  F1-Score:  {results['F1-Score']:.4f}")
    print(f"  ROC-AUC:   {results['ROC-AUC']:.4f}")
    
    return results


def save_model(model, model_name: str, output_dir: str):
    """
    Guarda un modelo entrenado en disco.
    
    Args:
        model: Modelo entrenado
        model_name: Nombre del modelo
        output_dir: Directorio de salida
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Limpiar nombre de archivo
    safe_name = model_name.replace(' ', '_').lower()
    filepath = os.path.join(output_dir, f'{safe_name}.joblib')
    
    joblib.dump(model, filepath)
    print(f"Modelo guardado: {filepath}")


def load_model(model_name: str, models_dir: str) -> Any:
    """
    Carga un modelo guardado.
    
    Args:
        model_name: Nombre del modelo
        models_dir: Directorio donde están los modelos
        
    Returns:
        Modelo cargado
    """
    safe_name = model_name.replace(' ', '_').lower()
    filepath = os.path.join(models_dir, f'{safe_name}.joblib')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Modelo no encontrado: {filepath}")
    
    return joblib.load(filepath)


def get_model_summary(model) -> Dict[str, Any]:
    """
    Obtiene información resumida de un modelo.
    
    Args:
        model: Modelo entrenado
        
    Returns:
        Diccionario con información del modelo
    """
    summary = {
        'type': type(model).__name__,
        'parameters': model.get_params()
    }
    
    # Si es un modelo de árbol, obtener importancia de features
    if hasattr(model, 'feature_importances_'):
        summary['feature_importances'] = model.feature_importances_
    
    # Si tiene coeficientes (regresión logística, SVM lineal)
    if hasattr(model, 'coef_'):
        summary['coefficients'] = model.coef_
    
    return summary


def compare_models(results_dict: Dict[str, Dict[str, float]], 
                  metric: str = 'F1-Score') -> pd.DataFrame:
    """
    Compara resultados de múltiples modelos.
    
    Args:
        results_dict: Diccionario {modelo: {métrica: valor}}
        metric: Métrica principal para ordenar
        
    Returns:
        DataFrame con comparación
    """
    df = pd.DataFrame(results_dict).T
    df = df.sort_values(metric, ascending=False)
    
    return df


def train_single_model(model_type: str, X_train, y_train, 
                      random_state: int = 42, **kwargs) -> Any:
    """
    Entrena un único modelo del tipo especificado.
    
    Args:
        model_type: Tipo de modelo ('logistic', 'rf', 'gb', etc.)
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        random_state: Semilla
        **kwargs: Parámetros adicionales para el modelo
        
    Returns:
        Modelo entrenado
    """
    models = get_models_dict(random_state)
    
    # Buscar modelo por nombre
    model = None
    for name, m in models.items():
        if model_type.lower() in name.lower():
            model = m
            break
    
    if model is None:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")
    
    # Aplicar parámetros adicionales
    if kwargs:
        model.set_params(**kwargs)
    
    # Entrenar
    model.fit(X_train, y_train)
    
    return model
