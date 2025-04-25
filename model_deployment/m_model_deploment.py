
def predict_popularity(duration, danceability, valence):
    import pandas as pd
    import joblib
    import os
    
    model = joblib.load(os.path.join(os.path.dirname(__file__), 'modelo_musica.pkl'))

    data = pd.DataFrame([{
        'duration_ms': duration,
        'danceability': danceability,
        'valence': valence
    }])

    return model.predict(data)[0]
