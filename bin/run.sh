#!/usr/bin/env bash
set -e

# Cambiar al directorio src relativo a este script
cd "$(dirname "$0")/../src"

# Si no existe entorno virtual, crearlo e instalar dependencias
if [ ! -d "venv" ]; then
    echo "⚙️  Setting up virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Ejecutar el recomendador
python3 recommender.py
