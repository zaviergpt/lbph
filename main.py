import os
import cv2
import json
import numpy
import base64
from flask import Flask, send_file
from flask_socketio import SocketIO

class Camera(Flask):

    def __init__(self):
        Flask.__init__(self, __name__)
        if not os.path.isfile("config.json"):
            with open("config.json", "w") as config:
                config.write(json.dumps({ "mode": "training" }))
                config.close()
        self.configuration = json.loads(open("config.json", "r").read())

    def start(self):
        io = SocketIO(self, cors_allowed_origins="*")
        recongizer = cv2.face.LBPHFaceRecognizer_create()
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        @io.on("image")
        def image(data):
            data = json.loads(data)
            frame = cv2.imdecode(numpy.frombuffer(base64.b64decode(data["data"].split(",")[1]), dtype=numpy.uint8), cv2.IMREAD_COLOR)
            print(frame.shape)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            io.emit("image", json.dumps({
                "id": data["id"],
                "data": "data:image/jpeg;base64,{}".format(base64.b64encode(cv2.imencode(".jpg", frame)[1]).decode("utf-8"))
            }))
        @self.route("/")
        def home():
            return send_file("./index.html")
        io.run(self, host="0.0.0.0", port=5000)

Camera().start()