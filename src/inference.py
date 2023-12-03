import numpy as np
from PIL import Image
import tensorflow as tf
from typing import Callable, Tuple, Dict
from tensorflow.keras.models import load_model


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
    return class_map, conf_score
