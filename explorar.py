import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# 1. Cargar los datos
df = pd.read_csv("train.csv").dropna(subset=['danceability', 'energy', 'tempo', 'valence', 'Class'])

# --- AQUÍ PIDES LA CANCIÓN POR TECLADO ---
mi_cancion = input("🎵 Escribe el nombre de la canción (ej. Hitch a Ride): ")

# Pequeña comprobación: ¿Existe esa canción en el CSV?
if mi_cancion not in df['Track Name'].values:
    print("❌ Vaya, esa canción no está en la base de datos o está mal escrita.")
else:
    # 2. EXTRAER EL GÉNERO DE ESA CANCIÓN AUTOMÁTICAMENTE
    cancion_usuario = df[df['Track Name'] == mi_cancion].iloc[0]
    genero_extraido = cancion_usuario['Class']

    print(f"\n✅ Canción encontrada: {cancion_usuario['Artist Name']} - {cancion_usuario['Track Name']}")
    print(f"🔍 Género detectado automáticamente: Clase {genero_extraido}")

    # 3. FILTRAR LA BASE DE DATOS POR ESE GÉNERO
    df_filtrado = df[df['Class'] == genero_extraido].copy()

    # 4. FÓRMULA MATEMÁTICA (Similitud)
    caracteristicas = ['danceability', 'energy', 'tempo', 'valence']
    scaler = StandardScaler()
    
    df_filtrado[caracteristicas] = scaler.fit_transform(df_filtrado[caracteristicas])
    todos_los_numeros = df_filtrado[caracteristicas].values
    numeros_usuario = df_filtrado[df_filtrado['Track Name'] == mi_cancion][caracteristicas].values

    df_filtrado['Similitud'] = cosine_similarity(numeros_usuario, todos_los_numeros)[0]

    # 5. DAR SIMILITUDES
    recomendaciones = df_filtrado.sort_values(by='Similitud', ascending=False)
    print("\n🔥 TOP 5 CANCIONES SIMILARES:")
    print(recomendaciones[['Artist Name', 'Track Name', 'Similitud']].iloc[1:6])