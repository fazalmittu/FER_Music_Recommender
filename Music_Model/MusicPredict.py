# from Music_Model.nn_musicclassifier import get_songs_features, predict_mood
# from Music_Model.helper_predict import make_predictions_on_playlist
# from Music_Model.helper_predict import get_songs_features
from os import name
from Music_Model.helper_predict import get_songs_features, predict_mood, get_playlist_data, make_predictions_on_playlist
import joblib
import numpy as np
import pandas as pd

# Anger, Fear - Energetic

# Surprised, Happy - Happy/Cheerful

# Neutral, Disgust - Chill/Calm

# Sad - Sad

facial_expressions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def recommend_song(emotion, playlist): 
    predictions = make_predictions_on_playlist(playlist)
    song_ids = predictions[0]
    song_emotion_list = predictions[1]
    user_emotion_list = []
    name_songs = []

    
    if emotion == 'Angry' or emotion == 'Fear':
        emotion_trigger = 'Energetic'
    elif emotion == 'Surprise' or emotion == 'Happy':
        emotion_trigger = 'Happy'
    elif emotion == 'Neutral' or emotion == 'Disgust':
        emotion_trigger = 'Calm'
    else:
        emotion_trigger = 'Sad'
    
    # print(song_emotion_list)
    # print(emotion_trigger)

    for i in range(len(song_emotion_list)):
        if song_emotion_list[i] == emotion_trigger:
            user_emotion_list.append(song_ids[i])
    
    for i in user_emotion_list:
        name_songs.append(get_songs_features(i)[0][0])
    
    return user_emotion_list

# print(recommend_song('Happy', 'https://open.spotify.com/playlist/2GUaZbnvUYD6CatRYxjoPJ?si=429498947e7a4a85'))




