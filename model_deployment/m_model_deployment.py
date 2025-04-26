import pandas as pd
import joblib
import sys
import os

def predict_popularity(duration, danceability, valence):
    model = joblib.load(os.path.dirname(__file__) + '/modelo_musica.pkl')
    
    data = pd.DataFrame([{
        'duration_ms': duration,
        'danceability': danceability,
        'valence': valence
    }])
    
    pred = model.predict(data)[0]
    return pred

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('Por favor ingresa: duration_ms, danceability, y valence')
    else:
        duration_ms = float(sys.argv[1])
        danceability = float(sys.argv[2])
        valence = float(sys.argv[3])
        
        result = predict_popularity(duration_ms, danceability, valence)
        
        print(f"Duración: {duration_ms} ms")
        print(f"Danceabilidad: {danceability}")
        print(f"Valencia: {valence}")
        print(f"Popularidad predicha (clase): {result}")
