import numpy as np
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
from utils import *

imgs_with_labels = []

for i in range(TRAIN_IMG_NUM):
    try:
        img = load_img(TRAIN_PATH + 'Cat/' + f'{i}.jpg')
        label = label_to_num('cat')
        imgs_with_labels.append((img, label))
    except Exception as e:
        print(e)

for i in range(TRAIN_IMG_NUM):
    try:
        img = load_img(TRAIN_PATH + 'Dog/' + f'{i}.jpg')
        label = label_to_num('dog')
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
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Steps per epoch: {len(X_train) // BATCH_SIZE}")
print(f"  Total epochs: {EPOCHS}")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
)

model.save(MODEL_NAME)
print(f'Model saved as \'{MODEL_NAME}\'')
