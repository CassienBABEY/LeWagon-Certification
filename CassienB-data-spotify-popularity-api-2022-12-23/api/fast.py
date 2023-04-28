


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../assets'))
loaded_model = joblib.load(f"{current_dir}/model.joblib")
app.state.model = loaded_model

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# define a root `/` endpoint
@app.get("/")
def index():
    return {"Status": "Up and running"}

### Je ne comprends pas pourquoi ça ne marche pas
###Edit: Note à moi même: Toujours vérifier les types

# Implement a /predict endpoint
@app.get("/predict")
def predict(acousticness: float, danceability: float,
            duration_ms: int, energy: float,
            explicit: int, id: str, instrumentalness: float,
            key: int, liveness: float, loudness: float,
            mode: int, name: str, release_date: str,
            speechiness: float, tempo: float, valence: float, artist: str):

    X_pred =pd.DataFrame(dict(
        acousticness= [float(acousticness)],
        danceability= [float(danceability)],
        duration_ms = [int(duration_ms)],
        energy = [float(energy)],
        explicit = [int(explicit)],
        id = [str(id)],
        instrumentalness = [float(instrumentalness)],
        key = [int(key)],
        liveness = [float(liveness)],
        loudness = [float(loudness)],
        mode = [int(mode)],
        name = [str(name)],
        release_date = [str(release_date)],
        speechiness = [float(speechiness)],
        tempo = [float(tempo)],
        valence = [float(valence)],
        artist = [str(artist)]
        ))


    # Use the model to predict the popularity of the song
    popularity = app.state.model.predict(X_pred)
    # Return the artist, name, and predicted popularity in a dictionary
    return {"artist": str(artist), "name": str(name), "popularity": float(popularity)}
