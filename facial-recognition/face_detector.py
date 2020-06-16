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

    def __guess_age(self, img):
        ageProto = "age_deploy.prototxt"
        ageModel = "age_net.caffemodel"
        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        ageNet = cv2.dnn.readNet(ageModel, ageProto)
        blob = cv2.dnn.blobFromImage(img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

    def __guess_gender(self, img):
        genderProto = "gender_deploy.prototxt"
        genderModel = "gender_net.caffemodel"
        genderList = ['Male', 'Female']
        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        genderNet = cv2.dnn.readNet(genderModel, genderProto)
        blob = cv2.dnn.blobFromImage(img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

    def get_frame(self, data):
        img = self.__read_image(data, cv2.IMREAD_COLOR)
        gray = self.__read_image(data, cv2.COLOR_BGR2GRAY)

        # Do magic!
        faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(10, 10),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        self.__guess_age(img)
        self.__guess_gender(img)

        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()
