# Celda temporal para recargar el módulo
import importlib
import sys

# Remover el módulo del caché si existe
if 'src.data_preprocessing' in sys.modules:
    del sys.modules['src.data_preprocessing']

# Reimportar
from src.data_preprocessing import create_target_variable

print("Módulo recargado exitosamente")
