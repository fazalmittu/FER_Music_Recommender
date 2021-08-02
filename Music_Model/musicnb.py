import joblib
from keras.models import load_model
import numpy as np

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

#Current credentials to access spotify API
#Enter cid and secret for spotipy API below
cid = '2a558674801e46b78ab77eaadb6b6b42'
secret = '0f6f356020614588a11e87e2e4953766'

# #Try to find way to get rid of credentials being publicly displayed [look up environment variables, write script to read it from input]

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

model = joblib.load('pip_model_UPDATED.pkl')
model.named_steps['keras'].model = load_model('keras_model_UPDATED.h5')

def get_songs_features(ids):
    meta = sp.track(ids)
    features = sp.audio_features(ids)

    # meta
    name = meta['name']
    album = meta['album']['name']
    artist = meta['album']['artists'][0]['name']
    release_date = meta['album']['release_date']
    length = meta['duration_ms']
    popularity = meta['popularity']
    ids =  meta['id']

    # features
    acousticness = features[0]['acousticness']
    danceability = features[0]['danceability']
    energy = features[0]['energy']
    instrumentalness = features[0]['instrumentalness']
    liveness = features[0]['liveness']
    valence = features[0]['valence']
    loudness = features[0]['loudness']
    speechiness = features[0]['speechiness']
    tempo = features[0]['tempo']
    key = features[0]['key']
    time_signature = features[0]['time_signature']

    # track = [danceability, acousticness, energy, instrumentalness, liveness, valence, loudness, speechiness, tempo]
    # columns = ['danceability','acousticness','energy','instrumentalness', 'liveness','valence','loudness','speechiness','tempo']
    # return track, columns

    track = [name, album, artist, ids, release_date, popularity, length, danceability, acousticness,
            energy, instrumentalness, liveness, valence, loudness, speechiness, tempo, key, time_signature]
    columns = ['name','album','artist','id','release_date','popularity','length','danceability','acousticness','energy','instrumentalness',
                'liveness','valence','loudness','speechiness','tempo','key','time_signature']
    return track,columns

def predict_mood(id_song):
    preds = get_songs_features(id_song)
    #Pre-process the features to input the Model
    preds_features = np.array(preds[0][7:-2]).reshape(-1,1).T
    
    print("Length of 1 array of features: ", len(preds_features[0]))

    #Predict the features of the song
    results = model.predict(preds_features)

    print(results)

    label_dict = {
        2:"Happy", 
        3:"Sad",
        0:"Calm",
        1:"Energetic"
    }

    mood = results
    name_song = preds[0][0]
    artist = preds[0][2]

    return print("{0} by {1} is a {2} song".format(name_song,artist,label_dict[mood[0]]))

predict_mood('0VjIjW4GlUZAMYd2vXMi3b')