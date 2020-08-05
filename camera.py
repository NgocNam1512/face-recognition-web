# import the necessary packages
import cv2
import numpy as np
import face_recognition
import time

class VideoCamera(object):
    def __init__(self):
       #capturing video
       self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        #releasing camera
        self.video.release()

    def get_frame(self, known_face_encodings, known_face_names):
        # Grab a single frame of video
        ret, frame = self.video.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.api.face_locations(rgb_small_frame)
        top, right, bottom, left = 0, 0, 0, 0
        for (top, right, bottom, left) in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        name = "Unknown"
        start = time.time()
        face_encodings = face_recognition.api.face_encodings(rgb_small_frame, face_locations, model='small')

        for face_encoding in face_encodings:
            matches = face_recognition.api.compare_faces(known_face_encodings, face_encoding)

            face_distances = face_recognition.api.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
        end = time.time()
        print(end-start)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 10), font, 1.0, (255, 255, 255), 1)
        
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()