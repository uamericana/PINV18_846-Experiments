import json
import logging

import numpy as np
import requests
import tensorflow as tf
from PIL import Image

logger = logging.getLogger(__name__)


def load_classes(json_file):
    with open(json_file, "r") as f:
        class_names = json.load(f)

    return class_names


def parse_image(filename, img_size):
    image = Image.open(filename).convert("RGB")
    image = np.asarray(image)
    image = tf.image.resize(image, img_size, method="bilinear")

    return image


class Resnetv2c3:
    def __init__(self, json_file, serving_base_url, serving_model_name):
        self.class_names = load_classes(json_file)
        self.img_size = (160, 160)
        self.model_name = serving_model_name
        self.serving_base_url = serving_base_url

    def classify_image(self, image_file):
        class_names = self.class_names
        image = parse_image(image_file, self.img_size)

        data = json.dumps({"signature_name": "serving_default", "instances": [image.numpy().tolist()]})
        headers = {"content-type": "application/json"}
        serving_url = f'http://{self.serving_base_url}/{self.model_name}:predict'
        json_response = requests.post(serving_url, data=data, headers=headers)
        prediction_logits = json.loads(json_response.text)['predictions']
        scores = tf.nn.softmax(prediction_logits)
        # Extract single result
        prediction_class_name = class_names[np.argmax(scores[0])]
        probs = [(class_names[i], score) for i, score in enumerate(scores.numpy()[0])]
        return prediction_class_name, probs
