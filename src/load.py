from tensorflow.keras.preprocessing import image
import xml.etree.ElementTree as et

IMG_NUM = 3686

def load_labeled_img(img_name: str) -> tuple[str, str]:
    img = image.load_img(f'archive/images/{img_name}.png')
    tree = et.parse(f'archive/annotations/{img_name}.xml')
    root = tree.getroot()
    label = root.find('.//object/name').text
    return (img, label)

for i in range(IMG_NUM):
    print(load_labeled_img(f'Cats_Test{i}'))