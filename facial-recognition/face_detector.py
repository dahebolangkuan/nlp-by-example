import cv2
import base64
import numpy as np

face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")


class FaceDetector:

    def get_frame(self, data):
        img = self.__read_image(data, cv2.IMREAD_COLOR)
        gray = self.__read_image(data, cv2.COLOR_BGR2GRAY)

        # Identify the face
        face = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(10, 10),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        age = self.__guess_age(img)
        gender = self.__guess_gender(img)

        # Frame the identified face and print the captured age and gender
        for (x, y, w, h) in face:
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
        print('Original Dimensions : ', img.shape)

        scale_factor = 0.6  # percent of original size
        width = int(img.shape[1] * scale_factor)
        height = int(img.shape[0] * scale_factor)
        dim = (width, height)

        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    @staticmethod
    def __guess_age(img):
        age_net = FaceDetector.__load_net(img, 'age_net.caffemodel', 'age_deploy.prototxt')
        age = FaceDetector.__predict_feature(age_net,
                                             ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53',
                                              '60-100'])
        print(f'Age: {age[1:-1]} years')
        return age

    @staticmethod
    def __guess_gender(img):
        gender_net = FaceDetector.__load_net(img, 'gender_net.caffemodel', 'gender_deploy.prototxt')
        gender = FaceDetector.__predict_feature(gender_net, ['Male', 'Female'])
        print(f'Gender: {gender}')
        return gender

    @staticmethod
    def __load_net(img, model, proto):
        model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)
        net = cv2.dnn.readNet(model, proto)
        blob = cv2.dnn.blobFromImage(img, 1.0, (227, 227), model_mean_values, swapRB=False)
        net.setInput(blob)
        return net

    @staticmethod
    def __predict_feature(net, values):
        predictions = net.forward()
        return values[predictions[0].argmax()]
