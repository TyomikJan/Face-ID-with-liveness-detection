# Standard package import:
import os
import time
import random
import win32gui
import win32con
import subprocess
from math import sin, cos, radians
import sqlite3
# From packages import:
import cv2
from datetime import datetime
import dlib
import pickle
import keyboard
import numpy as np
import signal
from colorama import init
from termcolor import colored
import argparse

connection = sqlite3.connect("db.sqlite")
crsr = connection.cursor()
def sql_command(name):
    crsr.execute(f'SELECT authorized FROM authorizations WHERE name == "{name}";')
# colorama
init()

ap = argparse.ArgumentParser()
ap.add_argument("-fv", "--face_video", type=str, required=False,
        help="video to authorize user")

args = vars(ap.parse_args())

# Directories declaration:
root_directory = os.path.dirname(os.path.abspath('__file__'))
cascades_directory = os.path.join(root_directory, 'Cascades\\')
faces_directory = os.path.join(root_directory, 'Faces\\')
if not os.path.exists(faces_directory):
    os.makedirs(faces_directory)
haar_directory = os.path.join(cascades_directory, 'Haar\\')
models_directory = os.path.join(root_directory, 'Models\\')
liveness_directory = os.path.join(root_directory, 'Liveness\\')
if not os.path.exists(models_directory):
    os.makedirs(models_directory)


# Recognizer declaration:
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Cascades object:
face_cascade = cv2.CascadeClassifier(haar_directory + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(haar_directory + 'haarcascade_eye.xml')
facial_landmarks_cascade = dlib.shape_predictor(cascades_directory + 'FacialLandmarksPredictor\\facial_landmarks.dat')

def align_face(input_feed, angle):
    if angle == 0: return input_feed
    height, width = input_feed.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 0.9)
    result = cv2.warpAffine(input_feed, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
    return result
##
def align_face_coords(pos, input_feed, angle):
    if angle == 0: return pos
    x = pos[0] - input_feed.shape[1] * 0.4
    y = pos[1] - input_feed.shape[0] * 0.4
    new_x = x * cos(radians(angle)) + y * sin(radians(angle)) + input_feed.shape[1] * 0.4
    new_y = -x * sin(radians(angle)) + y * cos(radians(angle)) + input_feed.shape[0] * 0.4
    return int(new_x), int(new_y), pos[2], pos[3]


# Calling model and pickle file:

labels = {}
face_recognizer.read(models_directory + 'FaceModel.yml')

with open(models_directory + 'Faces.pickle', 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

# Getting file creation time
if args['face_video']:
        video_file_ctime = os.path.getctime(args['face_video'])
        if time.strftime('%Y-%m-%d %H',time.localtime())!= datetime.fromtimestamp(video_file_ctime).strftime('%Y-%m-%d %H'):
            os.kill(os.getpid(), signal.SIGTERM)

live_capture = cv2.VideoCapture(args['face_video'])

cont_loop = True
# Start FaceID:
while (cont_loop == True):
    ret, color_feed = live_capture.read()
    ratio = 300.0 / color_feed.shape[1]
    dimensions = (300, int(color_feed.shape[0] * ratio))
    gray_feed = cv2.cvtColor(color_feed, cv2.COLOR_BGR2GRAY)
    face_in_feed = face_cascade.detectMultiScale(gray_feed, scaleFactor=1.3, minNeighbors=5)

    if len(face_in_feed) == 1:
        for angle in [0, -30, 30]:
            tilted_face = align_face(gray_feed, angle)
            face_in_feed = face_cascade.detectMultiScale(tilted_face, scaleFactor=1.3, minNeighbors=5)
            if len(face_in_feed):
                face_in_feed = [align_face_coords(face_in_feed[-1], gray_feed, -angle)]
                break

    if len(face_in_feed) == 0:
        pass
    elif len(face_in_feed) > 3:
        color_feed = cv2.GaussianBlur(color_feed, (25, 25), 3)
    else:
        # Facial Landmarks:
        for (x, y, w, h) in face_in_feed:
#
            r = int(w / 20)
            # Focusing on face:
            roi_gray_feed = gray_feed[y:y + h, x:x + w]
            # ID = Name of the face; Confidence = Accuracy*
            ID, confidence = face_recognizer.predict(roi_gray_feed)
            identified = 'Matched : {}%'.format(str(round(confidence)))
            if confidence >= 60 and confidence < 100:
                live_capture.release()
                cv2.destroyAllWindows()
                if len(face_in_feed) == 0:
                    pass
                elif len(face_in_feed) == 1:
                    live_capture.release()
                    cv2.destroyAllWindows()
                    # Liveness check
                    directory_person = (faces_directory + labels[ID])
                    directory_model = (directory_person+'\\liveness.model')
                    directory_le = (directory_person+'\\le.pickle')
                    video_file_path = args['face_video']
                    if args['face_video']:
                        os.system(f'cd {liveness_directory} && python -u liveness_demo.py --model {directory_model} --le {directory_le} --detector face_detector --name {labels[ID]} --video {video_file_path}')
                    if args['face_video']==None:
                        os.system(f'cd {liveness_directory} && python -u liveness_demo.py --model {directory_model} --le {directory_le} --detector face_detector --name {labels[ID]}')
                    sql_command(labels[ID])
                    user_auth = crsr.fetchall()

                    if user_auth[0][0]== 'YES':
                        print(f'Access Granted for {labels[ID]}!')

                    elif user_auth[0][0]== 'NO':
                       print(f'Attention! Swindler is trying to enter the system. Authorization denied.')

                    cont_loop=False
                    break

    # Output window:

    if cv2.waitKey(5) & 0xFF == int(27) or cont_loop==False:
        print(colored('[FEED]', 'green'), ' Ending Live Output Feed...', colored('\n[EXIT]', 'blue'), ' Live Feed stopped.')
        break

connection.close()
live_capture.release()
cv2.destroyAllWindows()
os.kill(os.getpid(), signal.SIGTERM)
