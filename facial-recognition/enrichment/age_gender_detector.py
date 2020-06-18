import cv2

class AgeGenderDetector:
    def __init__(self) -> None:
        self.model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)
        self.model = {
            "age": {
                "model": "./models/age_net.caffemodel",
                "proto": "./models/age_deploy.prototxt",
                "scale": ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"]
            },
            "gender": {
                "model": "./models/gender_net.caffemodel",
                "proto": "./models/gender_deploy.prototxt",
                "scale": ["Male", "Female"]
            },
        }

    def detect_age(self, img):
        return self.__predict_feature(img, self.model["age"])

    def detect_gender(self, img):
        return self.__predict_feature(img, self.model["gender"])

    def __load_net(self, img, model, proto):
        net = cv2.dnn.readNet(model, proto)
        blob = cv2.dnn.blobFromImage(img, 1.0, (227, 227), self.model_mean_values, swapRB=False)
        net.setInput(blob)
        return net
    
    def __predict_feature(self, img, model):
        net = self.__load_net(img, model["model"], model["proto"])
        predictions = net.forward()
        return model["scale"][predictions[0].argmax()]
