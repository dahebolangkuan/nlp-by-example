import cv2
import base64
import numpy as np

face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")


class FaceDetector():

    def __init__(self) -> None:
        self.video_capture = cv2.VideoCapture(0)

    def __del__(self):
        # releasing camera
        self.video_capture.release()

    def __read_image(self, data, mode):
        encoded_data = data.split(',')[1]
        img = np.fromstring(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(img, mode)
        print('Original Dimensions : ', img.shape)

        scale_factor = 0.6  # percent of original size
        width = int(img.shape[1] * scale_factor)
        height = int(img.shape[0] * scale_factor)
        dim = (width, height)

        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    def get_frame(self, data):
        img = self.__read_image(data, cv2.IMREAD_COLOR)
        gray = self.__read_image(data, cv2.IMREAD_GRAYSCALE)

        # Do magic!
        faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()
