# Facial Emotion Recognition
# This file allows to predict the emotion of a certain image

# The program uses the haarcascade model (haarcascade_frontface_default.xml) to detect faces in an image. It later uses the CNN VGG16 to predict the emotion.

# There are 7 types of emotions it can predict: Angry, Disgust, Fear, Happy, Sad, Surprise, Netural

# Note: In order for the model to detect the face(s), all parts of the face must be visible. Additionally, it is preferred that the face is # directly looking at the camera, with minimal tilt or angle. Covering the face with hands or titling the face so that all parts of the 
# face are not visible may not allow the model to detect the face.

# Libraries needed to be installed for this project (pip install <library>): tensorflow, opencv-python


# load json and create model
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy
import os
import numpy as np
import cv2

def runFER(filename):

    BASE_DIR = r'/Users/fazalmittu/FER_Music_Recommender/FER_Model/' # change this to your personal directory
    # setting constants and file names
    MODEL_NAME = 'vgg16' # using the VGG16 model
    MODEL_FILE = 'fer_' + MODEL_NAME

    MODEL_FILE_H5 = BASE_DIR + MODEL_FILE + '.h5'
    MODEL_FILE_JSON = BASE_DIR + MODEL_FILE + '.json'

    

    # creates user_test folder
    USER_TEST_PATH = r'/Users/fazalmittu/FER_Music_Recommender/static/uploads'
    # if not(os.path.exists(os.path.join(USER_TEST_PATH, 'uploads'))):
    #     os.mkdir(USER_TEST_PATH)
        # print('Directory created.')

    # asks user to upload picture to /sample_predict/user_test/
    # print('Please upload your picture to the following directory: \n' + USER_TEST_PATH)
    # asks for file name of the picture
    # img_file_name = input("Please enter the file name of the test image (ex. 'happy_face.jpg') ")
    # if os.path.isfile(os.path.join(USER_TEST_PATH, img_file_name)):
        # checks if the picture is in the folder
        # print('You have succesfully uploaded the picture:', img_file_name)

    # loading the model
    json_file = open(MODEL_FILE_JSON, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(MODEL_FILE_H5)
    # print("Loaded model from disk")

    # setting image resizing parameters
    WIDTH = 48
    HEIGHT = 48
    x=None
    y=None
    # setting labels
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # loading image
    full_size_image = cv2.imread(os.path.join(USER_TEST_PATH, filename))
    # print("Image Loaded")

    # detecting face(s)
    gray=cv2.cvtColor(full_size_image,cv2.COLOR_RGB2GRAY)
    face = cv2.CascadeClassifier('/Users/fazalmittu/FER_Music_Recommender/FER_Model/haarcascade_frontalface_default.xml')
    faces = face.detectMultiScale(gray, 1.3, 5)

    # print(faces) # prints coordinates of bounding box for each face

    # detecting emotion
    for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            
            if MODEL_NAME == 'vgg16': # convert 1d image to 3d for vgg16
                cropped_img = np.insert(cropped_img,1, 0 ,axis = 3)
                cropped_img = np.insert(cropped_img,2, 0 ,axis = 3)
                
            cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
            cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # predicting the emotion
            yhat= loaded_model.predict(cropped_img)
            cv2.putText(full_size_image, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            # print("Emotion: "+labels[int(np.argmax(yhat))])

    return labels[int(np.argmax(yhat))]
    # cv2.imshow('Emotion', full_size_image)

    # draw bounding box for faces
    # def draw_bounding_box(face_coordinates, image_array, color):
    #     x, y, w, h = face_coordinates
    #     cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)

    # for face in faces:
    #     draw_bounding_box(face, gray, (0, 255, 0))
        
    # rgb_image = cv2.cvtColor(full_size_image, cv2.COLOR_BGR2RGB)

    # bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # # store image with bounding box in /user_test/labeled/
    # if not(os.path.exists(os.path.join(USER_TEST_PATH, 'labeled'))):
    #     os.mkdir(os.path.join(USER_TEST_PATH, 'labeled'))

    # cv2.imwrite(os.path.join(USER_TEST_PATH, 'labeled', filename), bgr_image)

# print(runFER("pfp.jpeg"))