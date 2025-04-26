import pandas as pd
import joblib
import os

# Cargar modelo una sola vez
model_path = os.path.join(os.path.dirname(__file__), 'modelo_musica.pkl')
model = joblib.load(model_path)

# Diccionario para traducir la clase a texto
label_mapping = {
    0: "Baja popularidad",
    1: "Media popularidad",
    2: "Alta popularidad"
}

def predict_popularity(duration, danceability, valence):
    data = pd.DataFrame([{
        'duration_ms': duration,
        'danceability': danceability,
        'valence': valence
    }])
    
    pred_class = model.predict(data)[0]
    pred_label = label_mapping.get(pred_class, "Desconocido")
    
    return pred_label
