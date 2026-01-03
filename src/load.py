import xml.etree.ElementTree as et
import imageio.v3 as iio

IMG_NUM = 3686
IMG_SIZE = 224

imgs_with_labels = []


def label_to_num(label:str) -> int:
    if label == 'cat':
        return 0
    elif label == 'dog':
        return 1
    return -1


def load_labeled_img(img_name: str) -> tuple[str, int]:
    img = iio.imread(f'archive/images/{img_name}.png')
    tree = et.parse(f'archive/annotations/{img_name}.xml')
    root = tree.getroot()
    label = label_to_num(root.find('.//object/name').text)
    return (img, label)


for i in range(10):
    imgs_with_labels.append(load_labeled_img(f'Cats_Test{i}'))
