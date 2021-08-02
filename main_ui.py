from Music_Model.MusicPredict import recommend_song
from flask import Flask, flash, redirect, render_template, url_for, request

import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
import tensorflow as tf
import os
import cv2
import urllib.request
import os
from werkzeug.utils import secure_filename
import random

from FER_Model.FERSamplePredict import runFER
from Music_Model.MusicPredict import recommend_song
# from Music_Model.helper_predict import *

#USER NEEDS TO CHANGE THE DIRECTORY PATHS
 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    playlist_link = request.form.get('playlist_link')
    print(playlist_link)

    if playlist_link == "":
        playlist_link = "https://open.spotify.com/playlist/2GUaZbnvUYD6CatRYxjoPJ?si=97a0f23424eb468c"

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)

        FERresult = runFER(filename)
        
        song_id = "https://open.spotify.com/embed/track/" + random.choice(recommend_song(FERresult, playlist_link))
        
        #Takes about 35 seconds after user uploads image

        flash('Image successfully uploaded and displayed below')
        flash('The emotion detected: ' + FERresult)
        # flash('Playlist Link: ', playlist_link)
        return render_template('index.html', filename=filename, song_id=song_id)
    else:
        flash('Allowed image types are - png, jpg, jpeg')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
    app.run()

    # 3000 4000
    # 1024

