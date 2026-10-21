# Music Recommendation Information System (Source Code)

## Overview
Content-based music recommendation engine built in Python using `pandas` and `scikit-learn`. The system standardizes acoustic features (`danceability`, `energy`, `acousticness`, `instrumentalness`, `valence`, `tempo`) using `StandardScaler` and computes on-demand Cosine Similarity across a unified catalog of 21,950 unique tracks.

## Directory Structure
- `recommender.py`: Main application containing the `MusicRecommender` class and interactive CLI.
- `requirements.txt`: Required Python packages.
- `data/train.csv` & `data/test.csv`: Local acoustic feature datasets.

## Manual Setup & Execution

1. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
pip install -r requirements.txt
python3 recommender.py
---

### Paso 2: Limpiar archivos antiguos de prueba y proteger tus claves

En tu carpeta `~/SP` todavía tienes el archivo antiguo `explorar.py` (que ya sustituimos por `src/recommender.py`) y `spotify_test.py` (que contiene tu `CLIENT_SECRET` privado de Spotify). 

Ejecuta esto en tu terminal para limpiar el proyecto y actualizar el `.gitignore`:

```bash
# 1. Borrar el script antiguo que ya no se usa
rm -f explorar.py

# 2. Borrar los archivos residuales de Windows/WSL si siguen ahí
rm -f *:Zone.Identifier

# 3. Actualizar .gitignore para ignorar entornos virtuales en cualquier subcarpeta y el test de Spotify
cat << 'EOF' > .gitignore
# Virtual environments and Python cache
venv/
src/venv/
__pycache__/
src/__pycache__/

# Datasets and local tests with API keys
*.csv
spotify_test.py
*:Zone.Identifier
