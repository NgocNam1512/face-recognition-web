from flask import Flask, render_template, Response
from camera import VideoCamera
import face_recognition
import time
import os

app = Flask(__name__)

KNOWN_DIR = 'known/'

known_face_encodings = []
known_face_names = []

known_filename = os.listdir(KNOWN_DIR)
for filename in known_filename:
    start = time.time()
    image = face_recognition.api.load_image_file(KNOWN_DIR + filename)
    face_encoding = face_recognition.api.face_encodings(image)[0]
    known_face_names.append(filename.replace(".jpg", ""))
    known_face_encodings.append(face_encoding)
    print("Encoding known face:", time.time() - start)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        #get camera frame
        frame = camera.get_frame(known_face_encodings, known_face_names)
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)