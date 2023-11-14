from flask import Flask, render_template, request
from threading import Thread, Lock
import cv2
from deepface import DeepFace as dp
import time
import openai
import os
import cv2
from keras.preprocessing import image
import warnings
from deepface import DeepFace as dp
warnings.filterwarnings("ignore")
import time
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np

# Global variable for emotion
emotion = None
intrests = "Food, football, comedy, games"
extra = ""
lock = Lock()  # Lock for thread safety

class FlaskApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.setup_routes()
        self.f = 1

        # Set up OpenAI API credentials
        openai.api_key = "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"

    def setup_routes(self):
        @self.app.route("/")
        def index():
            return render_template("index.html")

        @self.app.route("/api", methods=["POST"])
        def api():
            message = request.json.get("message")
            response = self.send_to_openai(message)
            return response

    def ref_ext():
        global emotion,intrests
        global extra 
        extra = f"I am :{emotion}, My intrestes are {intrests}, use this data if you need and provide data, my chat:"
    def send_to_openai(self, message):
        global emotion
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"{extra}{message}"}]
        )

        if completion.choices[0].message!=None:
            print(f"{extra}{message}")
            return completion.choices[0].message        
        else :
            return 'Failed to Generate response!'

    def run(self):
        self.app.run()

class FaceRecognition:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = cv2.VideoCapture(0)

    def detect_emotion(self):
        global emotion
        while True:
            ret, test_img = self.cap.read()

            # Emotion detection and face recognition code
            gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            faces_detected = self.face_cascade.detectMultiScale(gray_img, 1.32, 5)

            for (x, y, w, h) in faces_detected:
                roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from image
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img_pixels = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0) / 255.0

                # Emotion analysis using DeepFace
                predictions = dp.analyze(test_img, actions="emotion", silent=True, enforce_detection=False)

                if predictions:
                    detected_emotion = predictions[0]["dominant_emotion"]

                    with lock:
                        emotion = detected_emotion
                    FlaskApp.ref_ext()
                    print(detected_emotion)

            # For demonstration, sleep for a while
            time.sleep(1)

    def run(self):
        self.detect_emotion()

if __name__ == "__main__":
    flask_app = FlaskApp()
    face_recognition = FaceRecognition()

    flask_thread = Thread(target=flask_app.run)
    face_thread = Thread(target=face_recognition.run)

    flask_thread.start()
    face_thread.start()