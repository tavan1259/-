import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Reshape
import json


def save_history(history, filename='history.json'):
    history_dict = history.history
    history_dict_for_json = {k: [float(i) for i in v] for k, v in history_dict.items()}
    with open(filename, 'w') as f:
        json.dump(history_dict_for_json, f)

# 1. Load and preprocess the dataset from CSV
def load_emnist_csv(train_csv_path, test_csv_path):
    # Load training data
    train_data = pd.read_csv(train_csv_path)
    X_train = train_data.iloc[:, 1:].values.astype('float32') / 255.0
    y_train = train_data.iloc[:, 0].values
    
    # Load testing data
    test_data = pd.read_csv(test_csv_path)
    X_test = test_data.iloc[:, 1:].values.astype('float32') / 255.0
    y_test = test_data.iloc[:, 0].values
    
    # Reshape the data
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


    # Convert labels to categorical
    y_train = to_categorical(y_train - 1, 26)
    y_test = to_categorical(y_test - 1, 26)
    
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_emnist_csv('data_set/emnist-letters-train.csv', 'data_set/emnist-letters-test.csv')

# 2. Build the LSTM model
def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Reshape((28, 28*1), input_shape=input_shape))  # Flatten channel into the sequence
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = build_model((28, 28, 1), 26)
model.summary()

# 3. Image Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    fill_mode='nearest'
)
datagen.fit(X_train)


for epoch in [8,32,128]:
    # 4. Train the model with augmented data
    history = model.fit(datagen.flow(X_train, y_train, batch_size=128),
                                epochs=epoch,
                                validation_data=(X_test, y_test),
                                verbose=1)
    save_history(history, "./lstm_epochs_"+str(epoch)+"_history.json")
    model.save("./lstm_epochs_"+str(epoch)+".h5")
