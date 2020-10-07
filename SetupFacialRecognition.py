# Standard import:
import os
import sys
import time
import shutil
import base64
from tkinter import *
from tkinter import StringVar
from tkinter import messagebox
from math import sin, cos, radians

# Downloaded packages import:
import cv2
import argparse
import dlib
import pickle
import PIL.Image
import numpy as np
import openpyxl
from colorama import init
from termcolor import colored
from PIL import Image, ImageTk
from imutils import build_montages
import imutils
import signal
import sqlite3

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-u", "--user", type=str, required=True,
        help="username")
ap.add_argument("-p", "--password", type=str, required=True,
        help="password")
ap.add_argument("-dr", "--directory_r", type=str, required=True,
        help="real directory video")
ap.add_argument("-df", "--directory_f", type=str, required=True,
        help="fake directory video")

args = vars(ap.parse_args())




connection = sqlite3.connect("db.sqlite")
crsr = connection.cursor()
def sql_command(arg,arg2):
    return crsr.execute(f'INSERT INTO authorizations (name,password) VALUES("{arg}","{arg2}");')

def sql_command_2(arg):
    return crsr.execute(f'SELECT password FROM authorizations WHERE name == "{arg}";')

init()

# Directories declaration:
root_directory = os.path.dirname(os.path.abspath(__file__))
cascades_directory = os.path.join(root_directory, 'Cascades\\')
faces_directory = os.path.join(root_directory, 'Faces\\')
if not os.path.exists(faces_directory):
    os.makedirs(faces_directory)
haar_directory = os.path.join(cascades_directory, 'Haar\\')
models_directory = os.path.join(root_directory, 'Models\\')
liveness_directory = os.path.join(root_directory, 'Liveness\\')
if not os.path.exists(models_directory):
    os.makedirs(models_directory)

# Cascades object:
face_cascade = cv2.CascadeClassifier(haar_directory + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(haar_directory + 'haarcascade_eye.xml')
facial_landmarks_cascade = dlib.shape_predictor(cascades_directory + 'FacialLandmarksPredictor\\facial_landmarks.dat')

# New profile:
# The below function is used for creating new profiles which will be later used for identification. While setting up all fields
# are made mandatory for avoiding any errors.
def create_new_profile():
    global directory_person
    global real_directory_photos
    #Starting liveness setup
    global user_name
    user_name = args["user"]
    user_password = args["password"]
    # UNCOMENT LATER
    #sql_command_2(user_name)
    #user_exists = crsr.fetchall()
    if 1:#user_exists[0][0] == user_password:
        global real_directory_photos
        fake_directory_videos = args["directory_f"]
        real_directory_videos = args["directory_r"]
        fake_directory_photos = (faces_directory + user_name + '\\dataset\\' + 'fake')
        real_directory_photos = (faces_directory + user_name + '\\dataset\\' + 'real')
        directory_photos = (faces_directory + user_name + '\\dataset')
        directory_person = (faces_directory + user_name)
        directory_model = (directory_person+'\\liveness.model')
        directory_le = (directory_person+'\\le.pickle')
        # ADDING USER TO THE DATABASE (ERASE LATER)
        sql_command(user_name,user_password)
        connection.commit()
        connection.close()

        # Working with CLI
        os.system(f'mkdir {directory_photos}')
        os.system(f'mkdir {real_directory_photos}')
        os.system(f'mkdir {fake_directory_photos}')
        os.system(f'copy NUL {directory_model}')
        os.system(f'copy NUL {directory_le}')
        print('Directories and files made!')

        os.system(f'cd {liveness_directory} && python -u gather_examples.py --input {fake_directory_videos} --output {fake_directory_photos} --detector face_detector --skip 1')
        os.system(f'cd {liveness_directory} && python -u gather_examples.py --input {real_directory_videos} --output {real_directory_photos} --detector face_detector --skip 1')
        print('Videos separated in frames!')
        os.system(f'cd {liveness_directory} && python -u train.py --dataset {directory_photos} --model {directory_model} --le {directory_le}')
        print('Model trained')
        #os.system(f'cd {liveness_directory} && python -u liveness_demo.py --model {directory_model} --le {directory_le} --detector face_detector')
        #print('Process finished!')
        # DELETING UNNECESSARY PHOTOS and VIDEOS
        #os.system(f'rd /s /q {real_directory_videos} && rd /s /q {fake_directory_videos}')
        #os.system(f'rd /s /q {directory_photos}')

# Create profile
create_new_profile()

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_id = 0
faces_array = {}
IDs = []
FaceList = []

# Saving face images:
for root, dirs, files in os.walk(real_directory_photos):
    for file in files:
        if file.endswith('png') or file.endswith('jpg') or file.endswith('jpeg') or file.endswith('JPG'):
            path = os.path.join(root, file)
            face_name = user_name
            if not face_name in faces_array:
                faces_array[face_name] = face_id
                face_id += 1

            ID = faces_array[face_name]
            face_image = PIL.Image.open(path).convert('L')
            face_numpy_array = np.array(face_image, 'uint8')
            face_in_folder = face_cascade.detectMultiScale(face_numpy_array, scaleFactor=1.3, minNeighbors=5)

            # Face dimensions:
            for (x, y, w, h) in face_in_folder:
                roi_face_image = face_numpy_array[y:y + h, x:x + w]
                FaceList.append(roi_face_image)
                IDs.append(ID)

# Saving pickle file:
with open(models_directory + 'Faces.pickle', 'wb') as f:
    pickle.dump(faces_array, f)

# Saving Trained model:
face_recognizer.train(FaceList, np.array(IDs))
face_recognizer.save(models_directory + 'FaceModel.yml')

# Deleting photos
#user_directory = faces_directory + '\\'+user_name
#os.system(f'erase {user_directory}\*.png')

os.kill(os.getpid(), signal.SIGTERM)
