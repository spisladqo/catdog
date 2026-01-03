from tensorflow.keras.preprocessing import image
import xml.etree.ElementTree as et

IMG_NUM = 3686

images = []

for i in range(IMG_NUM):
    img = image.load_img(f'archive/images/Cats_Test{i}.png')

    tree = et.parse(f'archive/annotations/Cats_Test{i}.xml')
    root = tree.getroot()
    name = root.find('.//object/name').text

    print(img, name)
