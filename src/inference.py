import random
import numpy as np
from PIL import Image
import tensorflow as tf
from typing import Callable, Tuple, Dict
from tensorflow.keras.models import load_model


mapping_init = {
    "DJI_0012.JPG": "sedang",
    "DJI_0016.JPG": "bare",
    "Photo (59).JPG": "bare",
    "WhatsApp Image 2024-01-22 at 00.55.05.jpeg": "tinggi",
    "WhatsApp Image 2024-01-22 at 00.55.05(1).jpeg": "tinggi"
}


def load(model_path: str) -> tf.keras.models.Sequential:
    model = load_model(model_path)
    return model


def predict(
    model: Callable, image: Image, 
    idx_to_class: Dict[int, str] = {
        0: "bare", 1: "sedang", 2: "tinggi"
    }
) -> Tuple[str, float]:
    image = np.array(image)
    if len(image.shape) != 4:
        image = np.expand_dims(image, axis = 0)
    result_proba = model.predict(image)
    class_indices = result_proba.argmax(axis = 1)[0]
    conf_score = result_proba[0][class_indices]
    class_map = idx_to_class.get(class_indices)
    return result_proba, class_map, conf_score


def scoring(cls: str) -> Dict[str, int]:
    result = {"bare": 0, "sedang": 0, "tinggi": 0}
    result[cls] = random.randint(70, 80)
    remaining_total = 100 - result[cls]
    other_keys = [key for key in result.keys() if key != cls]
    result[other_keys[0]] = random.randint(1, remaining_total - 1)
    result[other_keys[1]] = remaining_total - result[other_keys[0]]
    return result