import xml.etree.ElementTree as et
import cv2
from numpy import ndarray
import tensorflow as tf
from numpy import ndarray
from tensorflow import image


TEST_IMG_NUM = 3686
TRAIN_IMG_NUM = 12499
IMG_NORMSIZE = 224

TEST_PATH = 'testing_data/archive/'
TRAIN_PATH = 'training_data/archive/PetImages/'

MODEL_NAME = 'my_model.keras'


def label_to_num(label:str) -> int:
    if label == 'cat':
        return 0
    elif label == 'dog':
        return 1
    return -1


def load_label(xml_path: str) -> str:
    tree = et.parse(xml_path)
    root = tree.getroot()
    label = root.find('.//object/name').text
    return label


def load_img(img_path: str) -> ndarray:
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def preprocess_img(img: ndarray) -> ndarray:
    if img.shape[:2] != (IMG_NORMSIZE, IMG_NORMSIZE):
        img = image.resize(img, [IMG_NORMSIZE, IMG_NORMSIZE])

    if len(img.shape) == 2:
        img = tf.stack([img, img, img], axis=-1)
    elif img.shape[-1] == 1:
        img = tf.concat([img, img, img], axis=-1)
    elif img.shape[-1] == 4:
        img = img[:, :, :3]

    img = tf.cast(img, tf.float32)
    img = img / 255.0

    return img
