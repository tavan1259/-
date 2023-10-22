import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Reshape

def save_history(history, filename='history.json'):
    history_dict = history.history
    history_dict_for_json = {k: [float(i) for i in v] for k, v in history_dict.items()}
    with open(filename, 'w') as f:
        json.dump(history_dict_for_json, f)

# Load the dataset
train = pd.read_csv('data_set/emnist-letters-train.csv')
test = pd.read_csv('data_set/emnist-letters-test.csv')

# Reshape the dataset and preprocess
def preprocess_data(data):
    images = data.iloc[:, 1:].values.astype('float32')
    images = np.apply_along_axis(rotate, 1, images)
    images = images.reshape(images.shape[0], 28, 28, 1)
    labels = data.iloc[:, 0].values.astype('int32') - 1  # zero indexing
    return images, labels

def rotate(image):
    image = image.reshape(28, 28)
    return np.fliplr(np.rot90(image, 3)).reshape(784)

X_train, y_train = preprocess_data(train)
X_test, y_test = preprocess_data(test)

# Image Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.10,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(X_train)


model = Sequential([
    Reshape((28, 28), input_shape=(28, 28, 1)),
    LSTM(128, return_sequences=True),
    LSTM(128),
    Dense(26, activation='softmax')  # 26 letters
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

for epoch in [8,32,128,512]:
    history = model.fit(datagen.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test), epochs=epoch)
    save_history(history, "./lstm_epochs_"+str(epoch)+"_history.json")
    model.save("./lstm_epochs_"+str(epoch)+".h5")
