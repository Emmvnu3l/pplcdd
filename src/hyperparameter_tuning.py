"""
Módulo de optimización de hiperparámetros para el proyecto de análisis de cáncer de piel.

Contiene funciones para GridSearchCV y RandomizedSearchCV.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer


def get_param_grids() -> Dict[str, Dict[str, Any]]:
    """
    Obtiene los espacios de búsqueda de hiperparámetros para cada modelo.
    
    Returns:
        Diccionario {nombre_modelo: {parametro: [valores]}}
    """
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5],
            'subsample': [0.8, 1.0]
        },
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        },
        'Decision Tree': {
            'max_depth': [3, 5, 10, 15, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'criterion': ['gini', 'entropy']
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01]
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
    }
    
    return param_grids


def run_grid_search(model, param_grid: Dict, X_train, y_train, 
                   scorer=None, cv: int = 5, n_jobs: int = -1) -> Dict[str, Any]:
    """
    Ejecuta GridSearchCV para optimizar hiperparámetros.
    
    Args:
        model: Modelo base
        param_grid: Grilla de parámetros a probar
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        scorer: Métrica a optimizar (default: F1)
        cv: Número de folds para cross-validation
        n_jobs: Número de jobs paralelos
        
    Returns:
        Diccionario con resultados de la búsqueda
    """
    if scorer is None:
        scorer = make_scorer(f1_score, pos_label=1)
    
    print(f"Iniciando GridSearchCV con {len(param_grid)} parámetros...")
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True
    )
    
    # Ejecutar búsqueda
    grid_search.fit(X_train, y_train)
    
    results = {
        'best_model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': pd.DataFrame(grid_search.cv_results_),
        'search_method': 'GridSearchCV'
    }
    
    print(f"Mejores parámetros: {results['best_params']}")
    print(f"Mejor score CV: {results['best_score']:.4f}")
    
    return results


def run_randomized_search(model, param_distributions: Dict, X_train, y_train,
                         scorer=None, n_iter: int = 20, cv: int = 5,
                         random_state: int = 42, n_jobs: int = -1) -> Dict[str, Any]:
    """
    Ejecuta RandomizedSearchCV para optimizar hiperparámetros.
    
    Args:
        model: Modelo base
        param_distributions: Distribuciones de parámetros
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        scorer: Métrica a optimizar (default: F1)
        n_iter: Número de iteraciones
        cv: Número de folds
        random_state: Semilla aleatoria
        n_jobs: Número de jobs paralelos
        
    Returns:
        Diccionario con resultados de la búsqueda
    """
    if scorer is None:
        scorer = make_scorer(f1_score, pos_label=1)
    
    print(f"Iniciando RandomizedSearchCV con {n_iter} iteraciones...")
    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scorer,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True
    )
    
    # Ejecutar búsqueda
    random_search.fit(X_train, y_train)
    
    results = {
        'best_model': random_search.best_estimator_,
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'cv_results': pd.DataFrame(random_search.cv_results_),
        'search_method': 'RandomizedSearchCV'
    }
    
    print(f"Mejores parámetros: {results['best_params']}")
    print(f"Mejor score CV: {results['best_score']:.4f}")
    
    return results


def evaluate_tuned_model(model, X_test, y_test) -> float:
    """
    Evalúa un modelo optimizado en el conjunto de prueba.
    
    Args:
        model: Modelo optimizado
        X_test: Features de prueba
        y_test: Target de prueba
        
    Returns:
        F1-Score en test
    """
    from sklearn.metrics import f1_score
    
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred)
    
    return score


def compare_results(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compara los resultados de múltiples optimizaciones.
    
    Args:
        results_dict: Diccionario {nombre_modelo: resultados}
        
    Returns:
        DataFrame comparativo
    """
    comparison = []
    
    for model_name, results in results_dict.items():
        comparison.append({
            'Modelo': model_name,
            'Search_Method': results['search_method'],
            'CV_Score': results['best_score'],
            'Test_Score': results.get('test_score', np.nan),
            'Best_Params': str(results['best_params'])
        })
    
    df = pd.DataFrame(comparison)
    df = df.set_index('Modelo')
    df = df.sort_values('CV_Score', ascending=False)
    
    return df


def get_optimization_summary(results: Dict[str, Any]) -> str:
    """
    Genera un resumen de la optimización.
    
    Args:
        results: Resultados de optimización
        
    Returns:
        Resumen como string
    """
    summary = []
    summary.append("="*60)
    summary.append("RESULTADOS DE OPTIMIZACIÓN")
    summary.append("="*60)
    summary.append("")
    
    summary.append(f"Método: {results['search_method']}")
    summary.append(f"Mejor Score CV: {results['best_score']:.4f}")
    summary.append("")
    
    summary.append("Mejores Parámetros:")
    for param, value in results['best_params'].items():
        summary.append(f"  {param}: {value}")
    
    summary.append("")
    summary.append("="*60)
    
    return "\n".join(summary)


def save_best_params(results: Dict[str, Any], filename: str = 'best_params.txt'):
    """
    Guarda los mejores parámetros encontrados.
    
    Args:
        results: Resultados de optimización
        filename: Nombre del archivo
    """
    import os
    
    summary = get_optimization_summary(results)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, '..', 'results', 'reports')
    output_dir = os.path.normpath(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"Parámetros guardados: {filepath}")
