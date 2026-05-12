"""
Módulo de evaluación de modelos para el proyecto de análisis de cáncer de piel.

Contiene funciones para evaluación, comparación y visualización de resultados.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import glob
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (confusion_matrix, classification_report, 
                           matthews_corrcoef, cohen_kappa_score,
                           precision_recall_curve, average_precision_score,
                           roc_curve, auc)


def load_all_models(models_dir: str) -> Dict[str, Any]:
    """
    Carga todos los modelos guardados en el directorio.
    
    Args:
        models_dir: Directorio con los modelos
        
    Returns:
        Diccionario {nombre_modelo: modelo}
    """
    models = {}
    
    if not os.path.exists(models_dir):
        print(f"Directorio no encontrado: {models_dir}")
        return models
    
    # Buscar archivos .joblib
    model_files = glob.glob(os.path.join(models_dir, '*.joblib'))
    
    for filepath in model_files:
        try:
            model_name = os.path.basename(filepath).replace('.joblib', '')
            model_name = model_name.replace('_', ' ').title()
            
            models[model_name] = joblib.load(filepath)
            print(f"Cargado: {model_name}")
        except Exception as e:
            print(f"Error cargando {filepath}: {e}")
    
    return models


def calculate_advanced_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Calcula métricas avanzadas de evaluación.
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones
        
    Returns:
        Diccionario con métricas
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'Matthews_Correlation': matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0,
        'Cohen_Kappa': cohen_kappa_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0
    }
    
    return metrics


def create_confusion_matrix_plot(y_true, y_pred, model_name: str = '', 
                                 figsize: Tuple[int, int] = (8, 6)):
    """
    Crea una visualización de la matriz de confusión.
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones
        model_name: Nombre del modelo
        figsize: Tamaño de la figura
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Bajo Riesgo', 'Alto Riesgo'],
               yticklabels=['Bajo Riesgo', 'Alto Riesgo'],
               square=True, cbar=True)
    
    plt.title(f'Matriz de Confusión - {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    
    return plt.gcf()


def generate_evaluation_report(y_true, y_pred, model_name: str) -> str:
    """
    Genera un reporte de evaluación en formato texto.
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones
        model_name: Nombre del modelo
        
    Returns:
        Reporte como string
    """
    report = []
    report.append("="*60)
    report.append(f"EVALUACIÓN: {model_name}")
    report.append("="*60)
    report.append("")
    
    # Classification report
    report.append("Classification Report:")
    report.append("-"*40)
    report.append(classification_report(y_true, y_pred, 
                                      target_names=['Bajo Riesgo', 'Alto Riesgo']))
    
    # Métricas avanzadas
    metrics = calculate_advanced_metrics(y_true, y_pred)
    report.append("\nMétricas Avanzadas:")
    report.append("-"*40)
    for metric, value in metrics.items():
        report.append(f"  {metric}: {value:.4f}")
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    report.append("\nMatriz de Confusión:")
    report.append("-"*40)
    report.append(f"  Verdaderos Negativos: {cm[0,0]}")
    report.append(f"  Falsos Positivos: {cm[0,1]}")
    report.append(f"  Falsos Negativos: {cm[1,0]}")
    report.append(f"  Verdaderos Positivos: {cm[1,1]}")
    
    return "\n".join(report)


def compare_roc_curves(models: Dict, X_test, y_test, 
                      figsize: Tuple[int, int] = (10, 8)):
    """
    Compara las curvas ROC de múltiples modelos.
    
    Args:
        models: Diccionario de modelos
        X_test: Features de prueba
        y_test: Target de prueba
        figsize: Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    
    for name, model in models.items():
        # Obtener probabilidades
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_proba = model.decision_function(X_test)
        else:
            continue
        
        # Calcular ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Clasificador Aleatorio')
    plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
    plt.title('Curvas ROC Comparativas', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()


def generate_final_summary(all_results: Dict[str, Dict]) -> str:
    """
    Genera un resumen final del proyecto.
    
    Args:
        all_results: Diccionario con todos los resultados
        
    Returns:
        Resumen como string
    """
    summary = []
    summary.append("="*70)
    summary.append("RESUMEN FINAL DEL PROYECTO")
    summary.append("="*70)
    summary.append("")
    
    # Ranking de modelos
    summary.append("RANKING DE MODELOS (por F1-Score):")
    summary.append("-"*50)
    
    sorted_models = sorted(all_results.items(), 
                          key=lambda x: x[1].get('F1-Score', 0), 
                          reverse=True)
    
    for idx, (name, results) in enumerate(sorted_models, 1):
        f1 = results.get('F1-Score', 0)
        summary.append(f"{idx}. {name}: F1 = {f1:.4f}")
    
    summary.append("")
    summary.append("="*70)
    
    return "\n".join(summary)


def save_evaluation_report(report: str, filename: str = 'evaluation_report.txt'):
    """
    Guarda un reporte de evaluación en disco.
    
    Args:
        report: Contenido del reporte
        filename: Nombre del archivo
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(current_dir, '..', 'results', 'reports')
    reports_dir = os.path.normpath(reports_dir)
    
    os.makedirs(reports_dir, exist_ok=True)
    
    filepath = os.path.join(reports_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Reporte guardado: {filepath}")
