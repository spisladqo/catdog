import numpy as np
import tensorflow as tf
from numpy import ndarray
from tensorflow import image
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
from utils import *

imgs_with_labels = []

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

for i in range(1000):
    try:
        img = utils.load_img(TRAIN_PATH + 'Cat/' + f'{i}.jpg')
        label = utils.label_to_num('cat')
        imgs_with_labels.append((img, label))
    except Exception as e:
        print(e)

    try:
        img = utils.load_img(TRAIN_PATH + 'Dog/' + f'{i}.jpg')
        label = utils.label_to_num('dog')
        imgs_with_labels.append((img, label))
    except Exception as e:
        print(e)

imgs_with_labels = [(preprocess_img(i), l) for (i, l) in imgs_with_labels]

model = models.Sequential([
    Input(shape=(IMG_NORMSIZE, IMG_NORMSIZE, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

images, labels = zip(*imgs_with_labels)

images = np.array(images)
labels = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42
)


print(f"Starting training with:")
print(f"  Training samples: {len(X_train)}")
print(f"  Validation samples: {len(X_val)}")
print(f"  Batch size: {32}")
print(f"  Steps per epoch: {len(X_train) // 32}")
print(f"  Total epochs: {10}")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
)

model.save(MODEL_NAME)
print(f'Model saved as \'{MODEL_NAME}\'')

def predict_image(img_name):
    img = utils.load_img(img_name)
    img = preprocess_img(img)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]
    return "dog" if pred > 0.5 else "cat"


# count = 0
# for i in range(900, 1000):
#     name = f"Cats_Test{i}"
#     act = label_to_num(predict_image(name))
#     _, exp = load_labeled_img(name)

#     if act != exp:
#         print(f"Wrong in {name}")
#         count += 1

# print("Guessed wrong: ", count)