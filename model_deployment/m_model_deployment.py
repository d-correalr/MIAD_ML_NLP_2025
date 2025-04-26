import pandas as pd
import joblib
import os

# --- Cargar el modelo una sola vez ---
model_path = os.path.join(os.path.dirname(__file__), 'modelo_musica.pkl')
model = joblib.load(model_path)

# --- Función para categorizar popularidad ---
def categorizar_popularidad(score):
    if score <= 17:
        return "Baja popularidad"
    elif score <= 50:
        return "Media popularidad"
    else:
        return "Alta popularidad"

# --- Función para validar los inputs de la API ---
def validate_inputs(duration, acousticness, valence, speechiness, danceability):
    if not (133860 <= duration <= 5237295):
        raise ValueError("duration_ms fuera de rango permitido (133860 - 5237295)")
    if not (0.0 <= acousticness <= 0.996):
        raise ValueError("acousticness fuera de rango permitido (0.0 - 0.996)")
    if not (0.0 <= valence <= 0.995):
        raise ValueError("valence fuera de rango permitido (0.0 - 0.995)")
    if not (0.0 <= speechiness <= 0.965):
        raise ValueError("speechiness fuera de rango permitido (0.0 - 0.965)")
    if not (0.0 <= danceability <= 0.985):
        raise ValueError("danceability fuera de rango permitido (0.0 - 0.985)")

# --- Función principal para predecir ---
def predict_popularity(duration, acousticness, valence, speechiness, danceability):
    # Validar los inputs antes de predecir
    validate_inputs(duration, acousticness, valence, speechiness, danceability)

    # Crear el DataFrame de entrada
    data = pd.DataFrame([{
        'duration_ms': duration,
        'acousticness': acousticness,
        'valence': valence,
        'speechiness': speechiness,
        'danceability': danceability
    }])

    # Hacer la predicción
    predicted_score = model.predict(data)[0]
    predicted_label = categorizar_popularidad(predicted_score)

    # Retornar la respuesta
    return {
        "predicted_popularity_score": round(predicted_score, 2),
        "popularity_category": predicted_label
    }
