# USAGE
# python liveness_demo.py --model liveness.model --le le.pickle --detector face_detector

# import the necessary packages
from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import signal
import sqlite3
from datetime import datetime
# Directories declaration:
root_directory = os.path.dirname(os.path.abspath('__file__'))
db_directory = root_directory[:-9] + '\\db.sqlite'



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
        help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True,
        help="path to label encoder")
ap.add_argument("-d", "--detector", type=str, required=True,
        help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
ap.add_argument("-n", "--name", type=str, required=True,
        help="name of the user to authorize")
ap.add_argument("-v", "--video", type=str, required=False,
        help="video to authorize user")

args = vars(ap.parse_args())

# Getting file creation time
if args['video']:
        video_file_ctime = os.path.getctime(args['video'])
        if time.strftime('%Y-%m-%d %H',time.localtime())!= datetime.fromtimestamp(video_file_ctime).strftime('%Y-%m-%d %H'):
               os.kill(os.getpid(), signal.SIGTERM)


# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
        "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")
model = load_model(args["model"])
le = pickle.loads(open(args["le"], "rb").read())


# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
if args['video'] == None:
        vs = VideoStream(src=0)
        vs.start()
if args['video']:
        vs = cv2.VideoCapture(args['video'])

#Connecting to the DATABASE
# IMPORTANT CHANGE THIS to a personal one!
connection = sqlite3.connect(db_directory)
crsr = connection.cursor()

continue_loop=True
# loop over the frames from the video stream
while continue_loop:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 600 pixels

        grabbed,frame = vs.read()
        frame = imutils.resize(frame, width=600)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections
                if confidence > args["confidence"]:
                        # compute the (x, y)-coordinates of the bounding box for
                        # the face and extract the face ROI
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # ensure the detected bounding box does fall outside the
                        # dimensions of the frame
                        startX = max(0, startX)
                        startY = max(0, startY)
                        endX = min(w, endX)
                        endY = min(h, endY)

                        # extract the face ROI and then preproces it in the exact
                        # same manner as our training data
                        face = frame[startY:endY, startX:endX]
                        face = cv2.resize(face, (32, 32))
                        face = face.astype("float") / 255.0
                        face = img_to_array(face)
                        face = np.expand_dims(face, axis=0)

                        # pass the face ROI through the trained liveness detector
                        # model to determine if the face is "real" or "fake"
                        preds = model.predict(face)[0]
                        j = np.argmax(preds)
                        label = le.classes_[j]
                        if label == 'real' and preds[j] > .9994:
                                continue_loop=False
                                user_name = args["name"]
                                crsr.execute(f'UPDATE authorizations SET authorized = "YES" WHERE name == "{user_name}";')
                                connection.commit()
                                connection.close()
                                break


                        # draw the label and bounding box on the frame
                        if args['video'] == None:
                                label = "{}: {:.4f}".format(label, preds[j])
                                cv2.putText(frame, label, (startX, startY - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                cv2.rectangle(frame, (startX, startY), (endX, endY),
                                        (0, 0, 255), 2)

        # show the output frame and wait for a key press
        if args['video'] == None:
                cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q") or continue_loop==False:
            break

# do a bit of cleanup
print("[INFO] Person identified and authorized. Closing video stream...")
if args['video'] == None:
        cv2.destroyAllWindows()
        vs.stream.stream.release()
        vs.stop()
        vs.update()
if args['video']:
     cv2.destroyAllWindows()
     vs.release()
os.kill(os.getpid(), signal.SIGTERM)
