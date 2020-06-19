import cv2
import base64
import numpy as np

face_cascade = cv2.CascadeClassifier("./models/haarcascade_frontalface_alt2.xml")


class FaceDetector:

    def __init__(self, feature_detector) -> None:
        self.feature_detector = feature_detector

    def get_frame(self, data):
        img = self.__read_image(data, cv2.IMREAD_COLOR)
        gray = self.__read_image(data, cv2.COLOR_BGR2GRAY)

        # Identify the face
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(10, 10),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Guesstimate age and gender
        age = self.feature_detector.detect_age(img)
        gender = self.feature_detector.detect_gender(img)

        # Frame the identified face and print the captured age and gender
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            img = cv2.putText(
                img,  gender + ': ' + age,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()

    @staticmethod
    def __read_image(data, mode):
        encoded_data = data.split(',')[1]
        img = np.fromstring(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(img, mode)

        scale_factor = 0.6  # percent of original size
        width = int(img.shape[1] * scale_factor)
        height = int(img.shape[0] * scale_factor)
        dim = (width, height)

        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

