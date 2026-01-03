import tensorflow as tf
import numpy as np
from numpy import ndarray
from utils import MODEL_NAME, TEST_PATH, TEST_IMG_NUM, label_to_num, load_label, load_img, preprocess_img

model = tf.keras.models.load_model(MODEL_NAME)

images = []
exp_labels = []
act_labels = []

for i in range(TEST_IMG_NUM):
    try:
        img = load_img(TEST_PATH + f'images/Cats_Test{i}.png')
        label = label_to_num(load_label(TEST_PATH + f'annotations/Cats_Test{i}.xml'))
        images.append(img)
        exp_labels.append(label)
    except Exception as e:
        print(e)

images = [preprocess_img(i) for i in images]

images = np.array(images)
exp_labels = np.array(exp_labels)


def predict_image(img: ndarray, exp_label: int) -> bool:
    pred = model.predict(img)[0][0]
    if pred > 0.5:
        act_label = 'dog'
    else:
        act_label = 'cat'

    if exp_label == label_to_num(act_label):
        return True
    return False


wrong_count = 0
for i in range(TEST_IMG_NUM):
    try:
        img = images[i]
        img = np.expand_dims(img, axis=0)
        label = exp_labels[i]
        res = predict_image(img, label)
        if not res:
            print('Predicted wrong img with number:', i)
            wrong_count += 1
    except Exception as e:
        print(e)

print("Guessed wrong in total: ", wrong_count)