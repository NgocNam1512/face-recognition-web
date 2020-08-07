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
        _, frame = self.video.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.api.face_locations(rgb_small_frame)

        start = time.time()
        name = "Unknown"
        face_encodings = face_recognition.api.face_encodings(rgb_small_frame, face_locations, model='small')
        print("Time encoding:", time.time() - start)

        start = time.time()
        for face_encoding in face_encodings:
            matches = face_recognition.api.compare_faces(known_face_encodings, face_encoding)

            face_distances = face_recognition.api.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
        end = time.time()
        print("Match time:", end-start)

        start = time.time()
        top, right, bottom, left = 0, 0, 0, 0
        for (top, right, bottom, left) in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 10), font, 1.0, (255, 255, 255), 1)
        print("Draw time: ", time.time() - start)
        
        _, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()