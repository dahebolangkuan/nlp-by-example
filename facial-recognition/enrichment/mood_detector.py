from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2


class MoodDetector:
    def __init__(self) -> None:
        self.model = load_model("./models/model_v6_23.hdf5")

    def detect_mood(self, img):
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        data = image.img_to_array(pil_img)
        keras_img = image.array_to_img(data)
        return np.argmax(self.model.predict(keras_img))
