import face_recognition
import cv2

# known_image = face_recognition.load_image_file("known/obama.jpg")
# unknown_image = face_recognition.load_image_file("unknown")

# biden_encoding = face_recognition.face_encodings(known_image)[0]
# for unknown in unknown_image:
#     unknown_encoding = face_recognition.face_encodings(unknown)
#     results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
#     print(results)

image = face_recognition.api.load_image_file("known/obama.jpg")
# print(image.shape)
print(image)
face_locations = face_recognition.api.face_locations(image)
cv2.rectangle(image, (72, 26), (124, 78), (0,255,0), 3)
# print(face_locations)
cv2.imwrite('face_location.jpg', image)
cv2.waitKey()