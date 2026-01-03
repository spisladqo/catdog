import xml.etree.ElementTree as et
import imageio.v3 as iio
from numpy import ndarray


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
    img = iio.imread(img_path)
    return img
