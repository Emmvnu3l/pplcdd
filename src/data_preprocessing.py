"""
Módulo de preprocesamiento de datos para el proyecto de análisis de cáncer de piel.

Contiene funciones para carga, limpieza, transformación y preparación de datos.
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, List


def load_data(data_path: str) -> pd.DataFrame:
    """
    Carga el dataset desde la ruta especificada.
    
    Args:
        data_path: Ruta al archivo CSV
        
    Returns:
        DataFrame con los datos cargados
    """
    df = pd.read_csv(data_path)
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def load_processed_data() -> pd.DataFrame:
    """
    Carga el dataset procesado desde data/processed/.
    
    Returns:
        DataFrame procesado con target incluido
    """
    # Buscar ruta relativa desde notebooks/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_path = os.path.join(current_dir, '..', 'data', 'processed', 'df_processed.csv')
    processed_path = os.path.normpath(processed_path)
    
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Dataset procesado no encontrado en: {processed_path}")
    
    return pd.read_csv(processed_path)


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Obtiene las columnas numéricas del DataFrame.
    
    Args:
        df: DataFrame de entrada
        
    Returns:
        Lista de nombres de columnas numéricas
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    Obtiene las columnas categóricas del DataFrame.
    
    Args:
        df: DataFrame de entrada
        
    Returns:
        Lista de nombres de columnas categóricas
    """
    return df.select_dtypes(include=['object']).columns.tolist()


def create_target_variable(df: pd.DataFrame, 
                          signos_alarma: List[str] = None) -> pd.DataFrame:
    """
    Crea la variable target basada en signos de alarma.
    
    Args:
        df: DataFrame de entrada
        signos_alarma: Lista de columnas de signos de alarma
        
    Returns:
        DataFrame con columna 'diagnostico' añadida
    """
    if signos_alarma is None:
        signos_alarma = [
            'Ulceración o sangrado espontáneo',
            'Bordes elevados/irregulares',
            'Endurecimiento',
            'Cambio reciente de tamaño/color',
            'Costra persistente'
        ]
    
    df_copy = df.copy()
    
    # Verificar columnas existentes
    columnas_validas = [col for col in signos_alarma if col in df_copy.columns]
    
    if not columnas_validas:
        raise ValueError(f"No se encontraron columnas de signos de alarma. "
                        f"Disponibles: {df_copy.columns.tolist()}")
    
    # Codificar variables categóricas a numéricas
    for col in columnas_validas:
        mapeo = {
            'Sí': 1, 'Si': 1, 'Presente': 1, 'Yes': 1,
            'No': 0, 'Ausente': 0
        }
        
        # Obtener valores únicos para verificar si necesita mapeo
        valores_unicos = df_copy[col].dropna().unique()
        
        # Si hay valores string (Sí/No), aplicar mapeo
        if any(isinstance(v, str) for v in valores_unicos):
            df_copy[col] = df_copy[col].map(mapeo).fillna(0).astype(int)
        elif pd.api.types.is_string_dtype(df_copy[col]) or df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].map(mapeo).fillna(0).astype(int)
        else:
            df_copy[col] = df_copy[col].astype(int)
    
    # Crear target: 1 si tiene al menos un signo de alarma
    df_copy['diagnostico'] = (df_copy[columnas_validas].sum(axis=1) > 0).astype(int)
    
    print(f"Target creado. Distribución:\n{df_copy['diagnostico'].value_counts()}")
    print(f"Porcentaje clase 1: {(df_copy['diagnostico'].sum() / len(df_copy)) * 100:.2f}%")
    
    return df_copy


def encode_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Codifica variables categóricas usando LabelEncoder.
    
    Args:
        df: DataFrame de entrada
        
    Returns:
        Tuple de (DataFrame codificado, diccionario de encoders)
    """
    from sklearn.preprocessing import LabelEncoder
    
    df_encoded = df.copy()
    encoders = {}
    
    categorical_cols = get_categorical_columns(df_encoded)
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
    
    print(f"Columnas codificadas: {len(categorical_cols)}")
    return df_encoded, encoders


def preprocess_for_training(df: pd.DataFrame, target_col: str = 'diagnostico') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepara datos para entrenamiento: separa X e y, codifica categorías.
    
    Args:
        df: DataFrame completo
        target_col: Nombre de la columna target
        
    Returns:
        Tuple de (X, y)
    """
    # Separar target
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Codificar categóricas
    X_encoded, _ = encode_categorical_features(X)
    
    return X_encoded, y


def get_feature_info(df: pd.DataFrame) -> Dict:
    """
    Obtiene información sobre las características del dataset.
    
    Args:
        df: DataFrame de entrada
        
    Returns:
        Diccionario con información de features
    """
    numeric = get_numeric_columns(df)
    categorical = get_categorical_columns(df)
    
    info = {
        'total_columns': len(df.columns),
        'numeric_columns': len(numeric),
        'categorical_columns': len(categorical),
        'numeric_names': numeric,
        'categorical_names': categorical,
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    
    return info


def save_processed_data(df: pd.DataFrame, filename: str = 'df_processed.csv'):
    """
    Guarda el DataFrame procesado en data/processed/.
    
    Args:
        df: DataFrame a guardar
        filename: Nombre del archivo
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(current_dir, '..', 'data', 'processed')
    processed_dir = os.path.normpath(processed_dir)
    
    os.makedirs(processed_dir, exist_ok=True)
    
    output_path = os.path.join(processed_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"Dataset procesado guardado en: {output_path}")
